"""ROI selection must keep each person's wrist keypoints and reject ambiguity."""
import unittest
import numpy as np
from fast_selection import SinglePersonDetector

class Detector:
    def __init__(self, boxes):
        self.boxes=np.asarray(boxes, dtype=np.float32).reshape(-1,4)
    def run_human_detection(self, image, **kwargs):
        points=np.ones((len(self.boxes),17,3),dtype=np.float32)
        for i in range(len(points)):
            points[i,:,:2]=i*100+11
        return dict(boxes=self.boxes,keypoints=points)

class SelectionTests(unittest.TestCase):
    def test_single_person_keeps_original_coordinates(self):
        d=SinglePersonDetector(Detector([[40,20,90,150]]))
        result=d.run_human_detection(None)
        np.testing.assert_equal(result['boxes'],[[40,20,90,150]])
        self.assertEqual(result['keypoints'].shape,(1,17,3))

    def test_multi_person_selects_aligned_keypoints(self):
        d=SinglePersonDetector(Detector([[10,10,90,190],[210,10,290,190]]))
        d.roi=[200,0,300,200]
        result=d.run_human_detection(None)
        np.testing.assert_equal(result['boxes'],[[210,10,290,190]])
        np.testing.assert_equal(result['keypoints'][0,:,:2],111)

    def test_ambiguous_and_empty_do_not_fall_back_to_first(self):
        for boxes, roi in [([],None),([[0,0,100,200],[200,0,300,200]],None),
                           ([[0,0,100,200],[200,0,300,200]],[0,0,300,200]),
                           ([[0,0,100,200]],[400,0,500,200])]:
            with self.subTest(boxes=boxes,roi=roi):
                d=SinglePersonDetector(Detector(boxes)); d.roi=roi
                with self.assertRaises(ValueError):d.run_human_detection(None)

if __name__=='__main__':unittest.main()
