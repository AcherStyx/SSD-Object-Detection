import unittest
import numpy as np

from utils.bbox import *


class Test(unittest.TestCase):

    def test_iou(self):
        self.assertAlmostEqual(iou([10, 10, 2, 2], [10, 10, 2, 2]), 1, places=4)
        self.assertAlmostEqual(iou([10, 10, 1, 1], [20, 20, 1, 1]), 0, places=4)
        self.assertAlmostEqual(iou([10, 10, 2, 2], [10, 10, 4, 4]), 0.25, places=4)
        self.assertAlmostEqual(iou([10, 10, 0, 0], [20, 20, 0, 0]), 0, places=4)
        self.assertAlmostEqual(iou([10, 10, -1, -1], [10, 10, -1, -1]), 0, places=4)
        self.assertAlmostEqual(iou([10, 10, 2, 2], [11, 11, 2, 2]), 1 / 7, places=4)
        self.assertAlmostEqual(iou([10, 10, 6, 6], [13, 13, 2, 2]), 1 / (36 + 3), places=4)
        self.assertAlmostEqual(iou([10, -10, 1, 1], [10, -10, 1, 1]), 1, places=4)

    def test_iou_n(self):
        print(
            iou_n(tf.Variable([[10, 10, 2, 2], [10, 10, 1, 1], [10, 10, 2, 2]], dtype=tf.float32),
                  tf.Variable([[10, 10, 2, 2], [20, 20, 1, 1], [10, 10, 4, 4]], dtype=tf.float32))
        )

    def test_match_bbox(self):
        print("==========match bbox==========")
        dummy_default_box = np.array([[10, 10, 2, 2], [10, 10, 0.5, 0.5], [11, 11, 3, 3]], dtype=np.float32)
        dummy_target_box = np.array([[0, 10, 10, 1, 1], [1, 11, 11, 2, 2]], dtype=np.float32)
        match_bbox(dummy_target_box[:, 0], dummy_target_box[:, 1:], dummy_default_box)

        dummy_default_box = np.random.normal(size=(20, 4))
        dummy_target_box = np.random.normal(size=(2, 5))
        match_bbox(dummy_target_box[:, 0], dummy_target_box[:, 1:], dummy_default_box)

        dummy_default_box = np.array([[10, 10, 1, 1], [20, 20, 1, 1], [20, 20, 0.5, 0.5]])
        dummy_target_box = np.array([[0, 10, 10, 0.5, 0.5], [1, 20, 20, 1, 1], [2, 20, 20, 0.5, 0.5]])
        cls, loc, mask = match_bbox(dummy_target_box[:, 0], dummy_target_box[:, 1:], dummy_default_box)
        np.testing.assert_almost_equal(loc,
                                       dummy_target_box[:, 1:])
        dummy_default_box = np.array([[10, 10, 1, 1], [20, 20, 1.1, 1.1], [20, 20, 0.5, 0.5]])
        dummy_target_box = np.array([[0, 15, 15, 13, 13], [1, 15, 15, 14, 14]])
        cls, loc, mask = match_bbox(dummy_target_box[:, 0], dummy_target_box[:, 1:], dummy_default_box)
        np.testing.assert_almost_equal(loc,
                                       np.array([[15, 15, 14, 14], [15, 15, 13, 13], [0, 0, 0, 0]]))
        print(cls)


if __name__ == '__main__':
    unittest.main()
