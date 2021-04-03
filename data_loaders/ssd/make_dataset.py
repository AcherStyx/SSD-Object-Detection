__package__ = "data_loaders.ssd_adaptation"

import cv2
import numpy as np
import tensorflow as tf
import logging
from tqdm import tqdm

from utils.bbox import draw_bbox
from ..coco.make_dataset import COCODataLoader, coco_names, coco_colors

logger = logging.getLogger(__name__)


class SSDDataLoader:

    def __init__(self,
                 dataset_root,
                 shuffle_buffer=1,
                 dataset="COCO"):
        self._train_resize = (300, 300)
        self._shuffle_buffer = shuffle_buffer

        if dataset == "COCO":
            self._data_source_train, self._data_source_val = COCODataLoader(dataset_root=dataset_root,
                                                                            prefetch=1).get_dataset()
            self._transfer = self._coco2ssd
            self._names = coco_names
            self._colors = coco_colors
        else:
            raise ValueError

        self._train_set, self._val_set = self._load()

    def _coco2ssd(self, batch_data):
        image, cls, box = batch_data
        h, w, _ = image.shape
        image = cv2.resize(image, self._train_resize)

        # relative result
        scale = np.array([w, h, w, h])
        box /= scale

        return image, cls, box

    def _load(self):
        def data_iter(source):
            for batch_data in source.as_numpy_iterator():
                image, cls, box = self._transfer(batch_data)
                yield image, cls, box

        set_train = tf.data.Dataset.from_generator(generator=lambda: data_iter(self._data_source_train),
                                                   output_signature=(
                                                       tf.TensorSpec(shape=self._train_resize + (3,), dtype=tf.float32),
                                                       tf.TensorSpec(shape=(None,), dtype=tf.float32),
                                                       tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
                                                   )).shuffle(self._shuffle_buffer)

        set_val = tf.data.Dataset.from_generator(generator=lambda: data_iter(self._data_source_val),
                                                 output_signature=(
                                                     tf.TensorSpec(shape=self._train_resize + (3,), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None,), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
                                                 )).shuffle(self._shuffle_buffer)

        return set_train, set_val

    def get_dataset(self):
        return self._train_set, self._val_set

    def get_names_and_colors(self):
        return self._names, self._colors

    def draw_bbox(self, batch_data):
        image, cls, box = batch_data
        image = image.numpy()
        cls = cls.numpy()
        box = box.numpy()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        w_scale, h_scale = self._train_resize
        box *= (w_scale, h_scale, w_scale, h_scale)

        return draw_bbox(image, box, cls, self._names, self._colors)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    loader = SSDDataLoader("../../datasets/coco",
                           shuffle_buffer=1)

    for my_image, my_cls, my_box in tqdm(loader.get_dataset()[1]):
        print(my_cls, my_box)
        cv2.imshow("preview", loader.draw_bbox((my_image, my_cls, my_box)))
        cv2.waitKey(0)
