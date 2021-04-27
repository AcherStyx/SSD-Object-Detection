import cv2
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


def match_bbox(cls, bbox, default_box, thresh=0.5):
    target_cls, target_box, default_box = np.array(cls), np.array(bbox), np.array(default_box)
    target_box_origin, default_box_origin = target_box.copy(), default_box.copy()
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

    assert np.shape(iou_result) == (n_targets, n_defaults)
    while True:
        index = np.unravel_index(np.argmax(iou_result), np.shape(iou_result))
        if iou_result[index] <= thresh:
            # print("break iou:", iou_result[index])
            # print("max iou:", np.max(iou_result))
            break
        # mask[index] = True
        index_list.append(index)
        iou_result[:, index[1]] = 0.0

    # print("target number:", len(target_box_origin))
    # print("index list len: ", len(index_list))

    mask = np.zeros((n_defaults,), dtype=np.bool)
    labeled_boxes = np.zeros_like(default_box_origin, dtype=np.float32)
    labeled_cls = np.zeros((n_defaults,), dtype=np.int32)
    for index in index_list:
        mask[index[1]] = True
        labeled_boxes[index[1], :] = target_box_origin[index[0], :]
        labeled_cls[index[1]] = int(target_cls[index[0]])
    return labeled_cls, labeled_boxes, mask


def apply_anchor_box(origin_bbox, default_box):
    assert np.shape(origin_bbox) == np.shape(default_box)  # n * [x,y,w,h]
    # print(np.shape(origin_bbox))

    xy_relative = (origin_bbox[:, :2] - default_box[:, :2]) / default_box[:, 2:]
    wh_relative = np.log(np.maximum(origin_bbox[:, 2:], 1e-5) / np.maximum(default_box[:, 2:], 1e-5))

    return np.concatenate([xy_relative, wh_relative], axis=-1)


def draw_bbox(image, bbox, cls_label, cls_names, cls_color, scores=None, show_names=True):
    """
    draw bounding box on image
    @param image: image in tf.Tensor, numpy array or python list
    @param bbox: [[cx,cy,w,h],...] int value, corresponding to pixel
    @param cls_label: [cls1,cls2,...]
    @param cls_names: [name1,name2,...]
    @param cls_color: [color1,color2,...]
    @param scores:
    @param show_names:
    @return:
    """
    # check length
    if scores is not None:
        assert len(bbox) == len(cls_label) == len(scores)
    else:
        assert len(bbox) == len(cls_label)

    if isinstance(image, tf.Tensor):
        image = image.numpy()
    image_numpy = np.array(image)
    if image_numpy.dtype == np.float32 or image_numpy.dtype == np.float64:
        image_numpy *= 255
    image_numpy = image_numpy.astype(np.uint8)

    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
    for index, (cat, (cx, cy, w, h)) in enumerate(zip(cls_label, bbox)):
        cx, cy, w, h = np.clip((cx, cy, w, h), -10000, 10000)
        cv2.rectangle(image_numpy,
                      (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)),
                      cls_color[int(cat)], 2)
        if show_names:
            if scores is not None:
                label_string = cls_names[int(cat)] + " " + str(scores[index])
            else:
                label_string = cls_names[int(cat)]
            text_size = cv2.getTextSize(label_string, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)[0]
            cv2.rectangle(image_numpy, (int(cx - w / 2) - 1, int(cy - h / 2) - text_size[1] - 2),
                          (int(cx - w / 2) + text_size[0], int(cy - h / 2) - 1), cls_color[int(cat)], -1)
            cv2.putText(image_numpy, label_string, (int(cx - w / 2), int(cy - h / 2) - 2),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 0, 0), 1)

    return image_numpy
