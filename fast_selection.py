"""Select one detected person without discarding YOLO wrist keypoints."""
import numpy as np

class SinglePersonDetector:
    def __init__(self, detector):
        self.detector = detector
        self.roi = None
        self.selected_bbox = None
        self.selected_keypoints = None
        self.hand_warning = False

    def run_human_detection(self, image, **kwargs):
        self.selected_bbox = None
        self.selected_keypoints = None
        self.hand_warning = False
        result = self.detector.run_human_detection(image, **kwargs)
        if not isinstance(result, dict) or result.get('keypoints') is None:
            raise ValueError('Fast 需要 YOLO-Pose 人物框及关键点')
        boxes = np.asarray(result['boxes']).reshape(-1, 4)
        points = np.asarray(result['keypoints'])
        if len(boxes) == 0:
            raise ValueError('未检测到人物，请让人体完整入画或使用标准版')
        if len(points) != len(boxes) or not np.isfinite(boxes).all() or not np.isfinite(points).all():
            raise ValueError('检测器返回了无效人物框或关键点')
        candidates = np.arange(len(boxes))
        if self.roi is not None:
            roi = np.asarray(self.roi)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            inside = ((centers >= roi[:2]) & (centers <= roi[2:])).all(axis=1)
            intersection = np.maximum(0, np.minimum(boxes[:, 2:], roi[2:]) - np.maximum(boxes[:, :2], roi[:2])).prod(axis=1)
            areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
            roi_area = np.prod(roi[2:] - roi[:2])
            iou = intersection / np.maximum(areas + roi_area - intersection, 1e-6)
            candidates = candidates[inside | (iou >= .3)]
        if len(candidates) != 1:
            raise ValueError('人物选择不明确，请框选一人（框内只保留一个人物中心）')
        index = int(candidates[0])
        self.selected_bbox = boxes[index].tolist()
        self.selected_keypoints = points[index].tolist()
        self.hand_warning = bool((points[index, [9, 10], 2] < .3).any())
        return dict(result, boxes=boxes[index:index+1].copy(), keypoints=points[index:index+1].copy())
