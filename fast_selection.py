# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Select one detected person without discarding YOLO wrist keypoints."""
import numpy as np

class SinglePersonDetector:
    def __init__(self, detector):
        self.detector = detector
        self.roi = None
        self.selected_bbox = None
        self.selected_keypoints = None
        self.hand_warning = False
        self.video_key=None;self.selection_mask=None;self._cached_key=None;self._cached_result=None;self._claimed=set()
        self.reserved_boxes=[]

    def run_human_detection(self, image, **kwargs):
        self.selected_bbox = None
        self.selected_keypoints = None
        self.hand_warning = False
        if self.video_key is not None and self.video_key==self._cached_key:
            result=self._cached_result
        else:
            result = self.detector.run_human_detection(image, **kwargs)
            self._cached_key=self.video_key;self._cached_result=result;self._claimed=set()
        if not isinstance(result, dict) or result.get('keypoints') is None:
            raise ValueError('Fast 需要 YOLO-Pose 人物框及关键点')
        boxes = np.asarray(result['boxes']).reshape(-1, 4)
        points = np.asarray(result['keypoints'])
        if len(boxes) == 0:
            raise ValueError('未检测到人物，请让人体完整入画或使用标准版')
        if len(points) != len(boxes) or not np.isfinite(boxes).all() or not np.isfinite(points).all():
            raise ValueError('检测器返回了无效人物框或关键点')
        candidates = np.arange(len(boxes))
        if self.roi is not None and self.selection_mask is None:
            roi = np.asarray(self.roi)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            inside = ((centers >= roi[:2]) & (centers <= roi[2:])).all(axis=1)
            intersection = np.maximum(0, np.minimum(boxes[:, 2:], roi[2:]) - np.maximum(boxes[:, :2], roi[:2])).prod(axis=1)
            areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
            roi_area = np.prod(roi[2:] - roi[:2])
            iou = intersection / np.maximum(areas + roi_area - intersection, 1e-6)
            candidates = candidates[inside | (iou >= .3)]
        if self.selection_mask is not None:
            mask=self.selection_mask>0;mask_area=float(mask.sum());scores=[]
            for i,box in enumerate(boxes):
                x1,y1=np.maximum(np.floor(box[:2]),0).astype(int)
                x2,y2=np.minimum(np.ceil(box[2:]),[image.shape[1],image.shape[0]]).astype(int)
                intersection=float(mask[y1:y2,x1:x2].sum())
                scores.append(intersection/max(mask_area+max(0,x2-x1)*max(0,y2-y1)-intersection,1))
            reserved=set(self._claimed)
            for previous in self.reserved_boxes:
                previous=np.asarray(previous)
                intersection=np.maximum(0,np.minimum(boxes[:,2:],previous[2:])-np.maximum(boxes[:,:2],previous[:2])).prod(axis=1)
                area=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(axis=1)
                prev_area=np.maximum(0,previous[2:]-previous[:2]).prod()
                reserved.update(np.flatnonzero(intersection/np.maximum(area+prev_area-intersection,1e-6)>.8).tolist())
            available=[i for i in range(len(boxes)) if i not in reserved]
            available.sort(key=lambda i:scores[i],reverse=True)
            if not available or scores[available[0]]<.1:
                raise ValueError('掩膜没有对应的 YOLO 人物，请检查该帧分割')
            if len(available)>1 and scores[available[1]]>=scores[available[0]]*.85:
                raise ValueError('掩膜与多个人物匹配不明确，已跳过该人物帧')
            candidates=np.array(available[:1])
        if len(candidates) != 1:
            raise ValueError('人物选择不明确，请框选一人（框内只保留一个人物中心）')
        index = int(candidates[0])
        if self.video_key is not None:self._claimed.add(int(index))
        self.selected_bbox = boxes[index].tolist()
        self.selected_keypoints = points[index].tolist()
        self.hand_warning = bool((points[index, [9, 10], 2] < .3).any())
        return dict(result, boxes=boxes[index:index+1].copy(), keypoints=points[index:index+1].copy())
