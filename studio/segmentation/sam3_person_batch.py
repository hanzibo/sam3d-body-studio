# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Independent, audited text-person masks for single-person RGB-D experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from studio.core.gpu_lease import acquire_gpu_lease
from studio_paths import path


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    lease=acquire_gpu_lease('sam3')
    os.environ['HF_HUB_OFFLINE']='1';sys.path.insert(0,str(path('sam3_repo')))
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model=build_sam3_image_model(checkpoint_path=str(path('sam3_checkpoint')),
        bpe_path=str(path('sam3_repo')/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
        load_from_HF=False,device='cuda',compile=False,enable_inst_interactivity=False)
    processor=Sam3Processor(model)
    rows=[];tiles=[]
    for entry in json.loads(Path(args.audit).read_text()):
        sample=Path(entry['sample']);dest=out/f"pose-{entry['pose']:03d}";dest.mkdir(exist_ok=True)
        image=Image.open(sample/'rgb.png').convert('RGB');began=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=processor.set_text_prompt('person',processor.set_image(image))
        torch.cuda.synchronize()
        masks=state['masks'].detach().cpu().numpy().reshape(-1,image.height,image.width)>0
        scores=state['scores'].detach().float().cpu().numpy().reshape(-1)
        if not len(masks):raise ValueError(f'{sample.name}: no person mask')
        for i,m in enumerate(masks):Image.fromarray(m.astype('uint8')*255).save(dest/f'candidate-{i}.png')
        selected=int(np.argmax(masks.sum(axis=(1,2))))
        mask=masks[selected];Image.fromarray(mask.astype('uint8')*255).save(dest/'mask.png')
        row=dict(pose=entry['pose'],sample=str(sample),mask=str(dest/'mask.png'),
            rgb_sha256=hashlib.sha256((sample/'rgb.png').read_bytes()).hexdigest(),
            mask_sha256=hashlib.sha256((dest/'mask.png').read_bytes()).hexdigest(),
            method='sam3_text_person',prompt='person',score=float(scores[selected]),selected=selected,candidate_scores=scores.tolist(),candidate_pixels=masks.sum(axis=(1,2)).tolist(),pixels=int(mask.sum()),
            inference_seconds=time.perf_counter()-began,meaning='Independent automatic RGB mask, not anatomical ground truth')
        (dest/'source.json').write_text(json.dumps(row,indent=2));rows.append(row)
        arr=np.asarray(image).copy();arr[mask]=(arr[mask]*.55+np.array([40,220,120])*.45).astype('uint8')
        tile=Image.fromarray(arr);tile.thumbnail((400,240));ImageDraw.Draw(tile).text((8,8),f"Pose {entry['pose']} / {scores[selected]:.3f}",fill='white');tiles.append(tile)
        print(json.dumps(row),flush=True)
    sheet=Image.new('RGB',(1600,240*3),'#111827')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%4)*400,(i//4)*240))
    sheet.save(out/'contact.png');(out/'masks.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
