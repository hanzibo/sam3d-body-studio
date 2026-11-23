# SPDX-License-Identifier: GPL-3.0-only
"""Regenerate flat timing CSV and per-batch summaries without GPU/model loading."""
import csv
import json
from pathlib import Path
import numpy as np


def summarize(directory):
    directory=Path(directory)
    rows=[json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines() if line.strip()]
    fields=['source_index','timestamp','hand_count','detect_ms','reconstruct_ms','frame_with_io_ms','prepare_cpu_ms','backbone_gpu_ms','hand_decoder_including_mhr_gpu_ms','output_copy_ms']
    with (directory/'timings.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            report=row['report'];timings=report.get('reconstruction_timings',{})
            writer.writerow({field:row.get(field,report.get(field,timings.get(field,''))) for field in fields})
    groups={}
    for count in (0,1,2):
        group=[row['report'] for row in rows if row['report']['hand_count']==count]
        # Explicitly discard the first three occurrences of each batch size.
        warmed=group[3:]
        item=dict(frames=len(group),warmup_occurrences_excluded=min(len(group),3),measured_frames=len(warmed))
        if warmed:
            for field in ('detect_ms','reconstruct_ms','frame_with_io_ms'):
                values=[report[field] for report in warmed];item[field]=dict(mean=float(np.mean(values)),median=float(np.median(values)),p95=float(np.percentile(values,95)))
            item['gpu_stage_means_ms']={key:float(np.mean([r['reconstruction_timings'][key] for r in warmed if key in r['reconstruction_timings']]))
                                       for key in ('backbone_gpu_ms','hand_decoder_including_mhr_gpu_ms') if any(key in r['reconstruction_timings'] for r in warmed)}
        groups[str(count)]=item
    result=dict(groups_by_detected_hand_count=groups,
                scope='CPU wall includes detector preprocessing and IO as named; GPU stages nested within reconstruct, decoder includes MHR; excluded first 3 occurrences per hand count; no annotated detection/3D accuracy')
    (directory/'timing-groups.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    import sys
    for directory in sys.argv[1:]:print(json.dumps(summarize(directory),indent=2))
