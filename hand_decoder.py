# SPDX-License-Identifier: LicenseRef-SAM-3D-Body
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 hanzibo and contributors.
"""Inference-only schedule adapted from upstream PromptableDecoder.forward.

All decoder layers and the final prediction remain. Selected intermediate
predictions update feedback immediately; skipped layers do neither operation.
See LICENSES/SAM-3D-Body.txt.
"""
import torch
import ast
import inspect
import textwrap
from types import MethodType


def head_parameter_function(head):
    """Specialize the upstream parameter math before its MHR call, verbatim."""
    original=type(head).forward
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    function=tree.body[0]
    split=next((i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
                and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                and n.value.func.attr=='mhr_forward'),None)
    if split is None:raise RuntimeError('Upstream hand parameter contract changed')
    names=('global_rot_6d','global_rot_euler','pred_pose_cont','pred_pose_euler',
           'pred_shape','pred_scale','pred_hand','pred_face')
    function.body=function.body[:split]+[ast.Return(ast.Tuple([ast.Name(n,ast.Load()) for n in names],ast.Load()))]
    scope=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(original),'exec'),scope)
    return MethodType(scope['forward'],head)


class CompiledHandPose:
    """Compile both sides of MHR without nesting Dynamo around graph replay.

    Parameter head / camera head -> existing MHR graph -> coordinate conversion,
    camera projection and crop projection. Static topology stays outside graphs.
    """
    def __init__(self,model,mode):
        head=model.head_pose_hand
        self.head=head;self.faces=head.faces.detach().cpu().numpy()
        parameter_fn=head_parameter_function(head)
        def before(tokens,initial_pose,initial_camera):
            values=parameter_fn(tokens[:,0],initial_pose)
            camera=model.head_camera_hand(tokens[:,0],initial_camera)
            return values,camera
        def after(values,geometry,camera,batch):
            r6,rotation,continuous,pose,shape,scale,hand,face=values
            vertices,keypoints,joints,params,rotations=geometry
            keypoints=keypoints[:,:70]
            if vertices is not None:vertices[..., [1,2]]*=-1
            keypoints[..., [1,2]]*=-1
            if joints is not None:joints[..., [1,2]]*=-1
            output=dict(pred_pose_raw=torch.cat((r6,continuous),1),pred_pose_rotmat=None,
                        global_rot=rotation,body_pose=pose,shape=shape,scale=scale,hand=hand,face=face,
                        pred_keypoints_3d=keypoints,pred_vertices=vertices,pred_joint_coords=joints,
                        joint_global_rots=rotations,mhr_model_params=params,pred_cam=camera)
            output=model.camera_project_hand(output,batch)
            output['pred_keypoints_2d_cropped']=model._full_to_crop(batch,output['pred_keypoints_2d'],model.hand_batch_idx)
            return output
        self.before=torch.compile(before,mode=mode,dynamic=False,fullgraph=True)
        self.after=torch.compile(after,mode=mode,dynamic=False,fullgraph=True)

    def __call__(self,tokens,initial_pose,initial_camera,batch):
        values,camera=self.before(tokens,initial_pose,initial_camera)
        _,rotation,_,pose,shape,scale,hand,face=values
        geometry=self.head.mhr_forward(global_trans=torch.zeros_like(rotation),global_rot=rotation,
            body_pose_params=pose,hand_pose_params=hand,scale_params=scale,shape_params=shape,
            expr_params=face,do_pcblend=True,return_keypoints=True,return_joint_coords=True,
            return_model_params=True,return_joint_rotations=True)
        result=self.after(values,geometry,camera,batch)
        result['faces']=self.faces
        return result


def scheduled_forward(decoder, selected, token_embedding, image_embedding,
                      token_augment=None, image_augment=None, token_mask=None,
                      channel_first=True, token_to_pose_output_fn=None,
                      keypoint_token_update_fn=None, hand_embeddings=None,
                      hand_augment=None):
    if channel_first:
        image_embedding=image_embedding.flatten(2).permute(0,2,1)
        if image_augment is not None:image_augment=image_augment.flatten(2).permute(0,2,1)
        if hand_embeddings is not None:
            hand_embeddings=hand_embeddings.flatten(2).permute(0,2,1)
            hand_augment=hand_augment.flatten(2).permute(0,2,1)
            if len(hand_augment)==1:hand_augment=hand_augment.repeat(len(hand_embeddings),1,1)
    if not decoder.do_interm_preds:
        raise ValueError('Hand schedule requires intermediate prediction decoder')
    outputs=[]
    for index,layer in enumerate(decoder.layers):
        if hand_embeddings is None:
            token_embedding,image_embedding=layer(token_embedding,image_embedding,token_augment,image_augment,token_mask)
        else:
            token_embedding,image_embedding=layer(token_embedding,torch.cat((image_embedding,hand_embeddings),1),token_augment,torch.cat((image_augment,hand_augment),1),token_mask)
            image_embedding=image_embedding[:,:image_augment.shape[1]]
        if index in selected:
            pose=token_to_pose_output_fn(decoder.norm_final(token_embedding),prev_pose_output=outputs[-1] if outputs else None,layer_idx=index)
            outputs.append(pose)
            if decoder.keypoint_token_update:
                token_embedding,token_augment,_,_=keypoint_token_update_fn(token_embedding,token_augment,pose,index)
    out=decoder.norm_final(token_embedding)
    outputs.append(token_to_pose_output_fn(out,prev_pose_output=outputs[-1] if outputs else None,layer_idx=len(decoder.layers)-1))
    return out,outputs


def callback_inputs(callback):
    """Fail explicitly if upstream changes its callback contract."""
    scope=dict(zip(callback.__code__.co_freevars,(cell.cell_contents for cell in callback.__closure__ or ())))
    required={'init_pose','init_camera','batch','batch_size'}
    if not required <= scope.keys():raise RuntimeError('Upstream hand callback contract changed')
    size=scope['batch_size']
    return scope['init_pose'].view(size,-1),scope['init_camera'].view(size,-1),scope['batch']
