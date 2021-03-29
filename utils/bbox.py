import unittest
import tensorflow as tf
import numpy as np


def iou(bbox_1, bbox_2):
    """
    calculate intersection over union
    notice: cx,cy can be -1,-2,-3,...
    @param bbox_1: cx,cy,w,h
    @param bbox_2: cx,cy,w,h
    """
    cx_1, cy_1, w_1, h_1 = bbox_1[0], bbox_1[1], bbox_1[2], bbox_1[3]
    cx_2, cy_2, w_2, h_2 = bbox_2[0], bbox_2[1], bbox_2[2], bbox_2[3]
    area_1 = w_1 * h_1
    area_2 = w_2 * h_2

    cx_union_min = tf.maximum(cx_1 - w_1 / 2, cx_2 - w_2 / 2)
    cy_union_min = tf.maximum(cy_1 - h_1 / 2, cy_2 - h_2 / 2)
    cx_union_max = tf.minimum(cx_1 + w_1 / 2, cx_2 + w_2 / 2)
    cy_union_max = tf.minimum(cy_1 + h_1 / 2, cy_2 + h_2 / 2)

    union_area = tf.maximum(0.0, cx_union_max - cx_union_min) * tf.maximum(0.0, cy_union_max - cy_union_min)

    return union_area / (area_1 + area_2 - union_area + 1e-10)


def iou_n(n_bbox_1, n_bbox_2):
    cx_1, cy_1, w_1, h_1 = n_bbox_1[:, 0], n_bbox_1[:, 1], n_bbox_1[:, 2], n_bbox_1[:, 3]
    cx_2, cy_2, w_2, h_2 = n_bbox_2[:, 0], n_bbox_2[:, 1], n_bbox_2[:, 2], n_bbox_2[:, 3]
    area_1 = w_1 * h_1
    area_2 = w_2 * h_2

    cx_union_min = np.maximum(cx_1 - w_1 / 2, cx_2 - w_2 / 2)
    cy_union_min = np.maximum(cy_1 - h_1 / 2, cy_2 - h_2 / 2)
    cx_union_max = np.minimum(cx_1 + w_1 / 2, cx_2 + w_2 / 2)
    cy_union_max = np.minimum(cy_1 + h_1 / 2, cy_2 + h_2 / 2)

    union_area = np.maximum(1e-10, cx_union_max - cx_union_min) * np.maximum(1e-10, cy_union_max - cy_union_min)

    return union_area / (area_1 + area_2 - union_area + 1e-10)


def match_bbox(target_box, default_box, thresh=0.5):
    target_box_origin = target_box.copy()
    default_box_origin = default_box.copy()
    n_targets = target_box.shape[0]
    n_defaults = default_box.shape[0]

    assert n_targets <= n_defaults, "number of default boxes should greater than the number of targets"
    assert thresh > 0.0, "thresh should greater than zero"

    target_box = np.repeat(target_box, n_defaults, axis=0)
    default_box = np.repeat(np.expand_dims(default_box, axis=0), n_targets, axis=0)
    default_box = np.reshape(default_box, (n_defaults * n_targets, 4))

    iou_result = iou_n(target_box, default_box)
    iou_result = np.reshape(iou_result, (n_targets, n_defaults)).copy()

    index_list = []

    iou_result_copy = iou_result.copy()
    for _ in range(n_targets):
        index = np.unravel_index(np.argmax(iou_result_copy), np.shape(iou_result_copy))
        iou_result_copy[index[0], :] = 0.0
        iou_result_copy[:, index[1]] = 0.0
        index_list.append(index)
        iou_result[:, index[1]] = 0.0

    while True:
        index = np.unravel_index(np.argmax(iou_result_copy), np.shape(iou_result_copy))
        if iou_result[index] <= thresh:
            break
        # mask[index] = True
        index_list.append(index)
        iou_result[:, index[1]] = 0.0

    mask = np.zeros((n_defaults,), dtype=np.bool)
    labeled_boxes = np.zeros_like(default_box_origin, dtype=np.float32)
    for index in index_list:
        mask[index[1]] = True
        labeled_boxes[index[1], :] = target_box_origin[index[0], :]


if __name__ == '__main__':
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
            dummy_target_box = np.array([[10, 10, 1, 1], [11, 11, 2, 2]], dtype=np.float32)
            match_bbox(dummy_target_box, dummy_default_box)

            dummy_default_box = np.random.normal(size=(20, 4))
            dummy_target_box = np.random.normal(size=(2, 4))
            match_bbox(dummy_target_box, dummy_default_box)

    unittest.main()
