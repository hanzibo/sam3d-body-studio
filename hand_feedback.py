# SPDX-License-Identifier: LicenseRef-SAM-3D-Body
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 hanzibo and contributors.
# Parameter assembly adapted from SAM MHRHead.mhr_forward; feedback from
# SAM3DBody hand callbacks. See LICENSES/SAM-3D-Body.txt.
"""Inference-only hand feedback with exact-support, corrective-aware geometry.

The default retains six pose predictions and five feedback updates. Optional
reduced schedules change feedback; geometry uses exact dependencies.
"""
import copy
from functools import wraps
from types import SimpleNamespace
import torch
from torch import nn
import roma
from fast_feedback import feedback_values


class SelectedHandMHR(nn.Module):
    def __init__(self, mhr, mapping, surface_ids=None):
        super().__init__()
        from hand_acceleration import DeviceMHR,GraphMHR
        original=mhr.module.module if isinstance(mhr,GraphMHR) else mhr
        device=mhr.module if isinstance(mhr,GraphMHR) else DeviceMHR(original)
        ct=original.character_torch;nverts=len(ct.blend_shape.base_shape)
        if tuple(mapping.shape)!=(308,nverts+127):raise ValueError('Unsupported hand keypoint mapping')
        # Upstream keeps only right-hand 21 rows; all other MHR70 rows are zero.
        weights=mapping[21:42].detach()
        support=(weights!=0).any(dim=0)
        vertices=support[:nverts].nonzero().flatten();joints=support[nverts:].nonzero().flatten()
        self.surface=surface_ids is not None
        if self.surface:
            requested=torch.as_tensor(surface_ids,device=mapping.device,dtype=torch.long)
            if requested.ndim!=1 or not len(requested) or not bool(((requested>=0)&(requested<nverts)).all()):raise ValueError('Invalid hand surface IDs')
            vertices=torch.unique(torch.cat((vertices,requested)),sorted=True)
            self.register_buffer('export_indices',torch.searchsorted(vertices,requested))
        self.register_buffer('vertices',vertices);self.register_buffer('joints',joints)
        self.register_buffer('mapping',torch.cat((weights[:,vertices],weights[:,nverts+joints]),dim=1).contiguous())
        self.register_buffer('base_shape',ct.blend_shape.base_shape[vertices].detach().contiguous())
        self.register_buffer('shape_vectors',ct.blend_shape.shape_vectors[:,vertices].detach().contiguous())
        self.register_buffer('expression_vectors',original.face_expressions_model.shape_vectors[:,vertices].detach().contiguous())
        self.register_buffer('corrective_weight',device.corrective_weight)
        self.register_buffer('corrective_bias',device.corrective_bias)
        final=getattr(original.pose_correctives_model.pose_dirs_predictor,'2')
        rows=(vertices[:,None]*3+torch.arange(3,device=vertices.device)).flatten()
        self.register_buffer('corrective_output_weight',final.weight[rows].detach().contiguous())
        self.character=ct
        lbs=copy.deepcopy(ct.linear_blend_skinning)
        remap=torch.full((nverts,),-1,device=vertices.device,dtype=torch.long)
        remap[vertices]=torch.arange(len(vertices),device=vertices.device)
        compact=remap[lbs.vert_indices_flattened];keep=compact>=0
        lbs.skin_indices_flattened=lbs.skin_indices_flattened[keep].clone()
        lbs.skin_weights_flattened=lbs.skin_weights_flattened[keep].clone()
        lbs.vert_indices_flattened=compact[keep].clone();lbs.num_vertices=len(vertices)
        self.skinning=lbs;self.cache={};self.parity_mm={}
        self.spec=dict(vertices=len(vertices),original_vertices=nverts,joints=len(joints),keypoints=21,full_skeleton=127,final_correctives=True,intermediate_correctives=bool(len(vertices)),influences=int(keep.sum()))
        # Keep reference outside Module registration: immutable original model
        # already belongs to the final head. No second GPU model is created.
        object.__setattr__(self,'reference',original)
        self.reference_mapping=mapping

    def forward(self,shape,params,expression):
        ct=self.character
        joint_params=ct.model_parameters_to_joint_parameters(torch.cat((params,torch.zeros_like(shape)),dim=1))
        state=ct.joint_parameters_to_skeleton_state(joint_params)
        if not len(self.vertices):
            # Verified from the loaded mapping, not a hand-authored assumption:
            # these 21 points depend only on joints. Surface correctives cannot
            # affect them. Final geometry still uses full pose correctives.
            points=state[:,self.joints,:3]/100
            keypoints=(self.mapping@points.permute(1,0,2).flatten(1,2)).reshape(21,len(shape),3).permute(1,0,2)
            return torch.cat((torch.zeros_like(keypoints),keypoints,torch.zeros((len(shape),28,3),device=shape.device,dtype=shape.dtype)),dim=1)
        rotation=roma.euler_to_rotmat('xyz',joint_params.reshape(len(shape),-1,7)[:,2:,3:6])
        features=torch.cat((rotation[..., :,0],rotation[..., :,1]),dim=-1)
        features[:,:,0]-=1;features[:,:,4]-=1
        hidden=torch.relu(torch.nn.functional.linear(features.flatten(1,2),self.corrective_weight,self.corrective_bias))
        corrective=torch.nn.functional.linear(hidden,self.corrective_output_weight).reshape(len(shape),-1,3)
        rest=torch.einsum('nvd,bn->bvd',self.shape_vectors,shape)+self.base_shape
        rest=rest+torch.einsum('nvd,bn->bvd',self.expression_vectors,expression)+corrective
        verts=self.skinning(state,rest)/100
        points=torch.cat((verts,state[:,self.joints,:3]/100),dim=1)
        # Same regression ordering as upstream, using every nonzero dependency.
        keypoints=(self.mapping@points.permute(1,0,2).flatten(1,2)).reshape(21,len(shape),3).permute(1,0,2)
        keypoints=torch.cat((torch.zeros_like(keypoints),keypoints,torch.zeros((len(shape),28,3),device=shape.device,dtype=shape.dtype)),dim=1)
        return (verts[:,self.export_indices],keypoints,state) if self.surface else keypoints

    @torch.compiler.disable
    def inference(self,shape,params,expression,graph=True):
        signature=tuple((tuple(x.shape),x.dtype,x.device) for x in (shape,params,expression))
        if len(shape) not in self.parity_mm:
            verts,state=self.reference(shape,params,expression)
            points=torch.cat((verts/100,state[...,:3]/100),dim=1)
            ref=(self.reference_mapping[:70]@points.permute(1,0,2).flatten(1,2)).reshape(70,len(shape),3).permute(1,0,2)
            ref[:,:21]=0;ref[:,42:]=0
            candidate=self(shape,params,expression)
            if self.surface:
                reference=(verts[:,self.vertices[self.export_indices]]/100,ref,state)
                for actual,expected in zip(candidate,reference):torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                self.parity_mm[len(shape)]=float((reference[0]-candidate[0]).abs().max())*1000
            else:
                torch.testing.assert_close(candidate,ref,rtol=1e-5,atol=1e-6)
                self.parity_mm[len(shape)]=float((ref-candidate).abs().max())*1000
        if not graph or shape.device.type!='cuda':return self(shape,params,expression)
        cached=self.cache.get(signature)
        if cached is None:
            inputs=tuple(x.clone() for x in (shape,params,expression));stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self(*inputs)
            torch.cuda.current_stream().wait_stream(stream)
            capture=torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture,stream=stream):out=self(*inputs)
            cached=(inputs,capture,out);self.cache[signature]=cached
        inputs,capture,out=cached
        for target,value in zip(inputs,(shape,params,expression)):target.copy_(value)
        capture.replay();return tuple(x.clone() for x in out) if self.surface else out.clone()


def selected_mhr_forward(head,selected,graph,global_trans,global_rot,body_pose_params,
                         hand_pose_params,scale_params,shape_params,expr_params=None,
                         **kwargs):
    """Preserve upstream hand-to-body parameter assembly and hand masking."""
    rotation=roma.rotmat_to_euler('xyz',roma.euler_to_rotmat('xyz',global_rot)@head.local_to_world_wrist)
    translation=-(roma.euler_to_rotmat('xyz',rotation)@(head.right_wrist_coords-head.root_coords)+head.root_coords)+global_trans
    scales=head.scale_mean[None,:]+scale_params@head.scale_comps
    pose=torch.cat((translation*10,rotation,body_pose_params[...,:130]),dim=1)
    if hand_pose_params is not None:pose=head.replace_hands_in_pose(pose,hand_pose_params)
    params=torch.cat((pose,scales),dim=1);params[:,head.nonhand_param_idxs]=0
    result=selected.inference(shape_params,params,expr_params,graph)
    if selected.surface:
        verts,keypoints,state=result
        return verts,keypoints,state[...,:3]/100,params,roma.unitquat_to_rotmat(state[...,3:7])
    return None,result,None,params,None


class HandFeedbackCall:
    def __init__(self,model,image,tokens,values_fn=feedback_values):
        self.view=SimpleNamespace(**{key:getattr(model,key+'_hand') for key in (
            'keypoint_embedding_idxs','keypoint3d_embedding_idxs','keypoint_posemb_linear',
            'keypoint_feat_linear','keypoint3d_posemb_linear')},pelvis_idx=model.pelvis_idx,cfg=model.cfg)
        self.image=image;self.values_fn=values_fn;self.n2=len(model.keypoint_embedding_idxs_hand);self.n3=len(model.keypoint3d_embedding_idxs_hand)
        self.start3=tokens-self.n3;self.start2=self.start3-self.n2
        self.pose=None;self.values=None;self.computations=0;self.hits=0

    def __call__(self,tokens,augment,pose,layer_idx):
        if pose is not self.pose:
            self.values=self.values_fn(self.view,self.image,pose);self.pose=pose;self.computations+=1
        else:self.hits+=1
        delta,aug2,aug3=self.values
        out=tokens.clone();out[:,self.start2:self.start3]+=delta
        aug=augment.clone();aug[:,self.start2:self.start3]=aug2;aug[:,self.start3:self.start3+self.n3]=aug3
        return out,aug,pose,layer_idx


def install_hand_feedback(model,config,vertex_ids=None):
    head=model.head_pose_hand;decoder=model.decoder_hand
    controller=dict(enabled=True,feedback=config.feedback_cache,keypoints=config.intermediate_keypoints,final_vertices=config.final_hand_vertices,last=None,version='exact_support_correctives_v1')
    selected=None;original_mhr=head.mhr_forward;state=dict(intermediate=False,final=False)
    if config.intermediate_keypoints:
        selected=SelectedHandMHR(head.mhr,head.keypoint_mapping).eval();head._studio_selected_hand=selected
        controller['geometry']=selected.spec
        def slim(*args,**kwargs):return selected_mhr_forward(head,selected,config.mhr_graph,*args,**kwargs)
        core=torch.compile(slim,mode=config.compile_mode,dynamic=False,fullgraph=False) if config.mhr_compile else slim
    final_core=None
    if config.final_hand_vertices:
        if vertex_ids is None:raise ValueError('Final hand surface requires explicit topology vertex IDs')
        final=SelectedHandMHR(head.mhr,head.keypoint_mapping,vertex_ids).eval()
        head._studio_final_hand=final;controller['final_geometry']=final.spec
        def compact(*args,**kwargs):return selected_mhr_forward(head,final,config.mhr_graph,*args,**kwargs)
        final_core=torch.compile(compact,mode=config.compile_mode,dynamic=False,fullgraph=False) if config.mhr_compile else compact
    if final_core is not None:head._studio_final_geometry=final_core
    if selected is not None or final_core is not None:
        @wraps(original_mhr)
        def mhr(*args,**kwargs):
            if state['intermediate']:return core(*args,**kwargs)
            if state['final']:return final_core(*args,**kwargs)
            return original_mhr(*args,**kwargs)
        head.mhr_forward=mhr
    values_fn=torch.compile(feedback_values,mode=config.compile_mode,dynamic=False,fullgraph=True) if config.decoder_compile else feedback_values
    from hand_decoder import CompiledHandPose,callback_inputs,scheduled_forward
    if len(decoder.layers)!=6:raise ValueError('Hand prediction schedule expects six decoder layers')
    controller.update(callback_compile=config.intermediate_compile,intermediate_layers=list(range(config.intermediate_predictions)),compiled_regions=['parameter_and_camera_heads','geometry_postprocess_and_projection'] if config.intermediate_compile else [],geometry_boundary='existing MHR compile / CUDA Graph')
    compiled_pose=CompiledHandPose(model,config.compile_mode) if config.intermediate_compile else None
    original=decoder.forward
    @wraps(original)
    def forward(*args,**kwargs):
        controller['last']=None
        if not controller['enabled'] or model.training or torch.is_grad_enabled():return original(*args,**kwargs)
        pose_fn=kwargs['token_to_pose_output_fn']
        inputs=callback_inputs(pose_fn) if controller['callback_compile'] else None
        counts=dict(intermediate_keypoint_calls=0,full_mhr_calls=0,compact_surface_calls=0)
        def pose(*pa,**pk):
            layer=pk.get('layer_idx',pa[2] if len(pa)>2 else None)
            state['intermediate']=config.intermediate_keypoints and controller['keypoints'] and layer!=len(decoder.layers)-1
            state['final']=config.final_hand_vertices and controller['final_vertices'] and layer==len(decoder.layers)-1
            counts['intermediate_keypoint_calls' if state['intermediate'] else ('compact_surface_calls' if state['final'] else 'full_mhr_calls')]+=1
            try:
                if controller['callback_compile']:
                    tokens=pa[0] if pa else pk['tokens']
                    return compiled_pose(tokens,*inputs)
                return pose_fn(*pa,**pk)
            finally:state.update(intermediate=False,final=False)
        kwargs['token_to_pose_output_fn']=pose
        call=None
        if config.feedback_cache and controller['feedback']:
            tokens=args[0] if args else kwargs['token_embedding'];image=args[1] if len(args)>1 else kwargs['image_embedding']
            call=HandFeedbackCall(model,image,tokens.shape[1],values_fn);kwargs['keypoint_token_update_fn']=call
        try:
            selected=controller['intermediate_layers']
            if selected==list(range(len(decoder.layers)-1)):return original(*args,**kwargs)
            return scheduled_forward(decoder,set(selected),*args,**kwargs)
        finally:
            state.update(intermediate=False,final=False)
            controller['last']=dict(**counts,feedback_computations=call.computations,feedback_hits=call.hits) if call else counts
    decoder.forward=forward;model._studio_hand_feedback=controller
    return controller
