__package__ = "data_loaders.ssd_adaptation"

import cv2
import numpy as np
import tensorflow as tf

from ..coco.make_dataset import COCODataLoader

coco_names = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
    'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class SSDDataset:

    def __init__(self,
                 dataset_root,
                 batch_size=1,
                 shuffle_buffer=1,
                 train_size=(300, 300),
                 dataset="COCO"):
        self._train_resize = train_size
        self._batch_size = batch_size
        self._shuffle_buffer = shuffle_buffer

        if dataset == "COCO":
            self._data_source_train, self._data_source_val = COCODataLoader(dataset_root=dataset_root,
                                                                            shuffle_buffer=1,
                                                                            prefetch=1).get_dataset()
            self._transfer = self._coco2ssd
        else:
            raise ValueError

        self._train_set, self._val_set = self._load()

    def _coco2ssd(self, batch_data):
        image, target = batch_data
        target = target.numpy()
        h, w, _ = image.shape

        image = tf.image.resize(image, self._train_resize)

        scale = np.array([w, h, w, h])

        # w,h->x_max,y_max
        target[:, 3:] += target[:, 1:3]
        target[:, 1:] /= scale

        return image, target

    def _load(self):
        def data_iter(source):
            for batch_data in source:
                yield self._transfer(batch_data)

        set_train = tf.data.Dataset.from_generator(generator=lambda: data_iter(self._data_source_train),
                                                   output_signature=(
                                                       tf.TensorSpec(shape=self._train_resize + (3,), dtype=tf.float32),
                                                       tf.TensorSpec(shape=(None, 5), dtype=tf.float32)
                                                   )).shuffle(self._shuffle_buffer).batch(self._batch_size)

        set_val = tf.data.Dataset.from_generator(generator=lambda: data_iter(self._data_source_val),
                                                 output_signature=(
                                                     tf.TensorSpec(shape=self._train_resize + (3,), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, 5), dtype=tf.float32)
                                                 )).shuffle(self._shuffle_buffer).batch(self._batch_size)

        return set_train, set_val

    def get_dataset(self):
        return self._train_set, self._val_set

    def draw_bbox(self, batch_data):
        image, target = batch_data
        image = image[0].numpy()
        target = target[0].numpy()

        h, w = self._train_resize

        for cls, x_min, y_min, x_max, y_max in target:
            cv2.rectangle(image, (int(x_min * w), int(y_min * h)), (int(x_max * w), int(y_max * h)), (255, 255, 255))
        return image


if __name__ == '__main__':
    loader = SSDDataset("../../datasets/coco")
    for my_image, my_target in loader.get_dataset()[0]:
        # print(my_image, my_target)
        cv2.imshow("preview", loader.draw_bbox((my_image, my_target)))
        cv2.waitKey(1)
