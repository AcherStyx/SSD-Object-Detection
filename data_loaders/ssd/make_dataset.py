__package__ = "data_loaders.ssd_adaptation"

import cv2
import numpy as np
import tensorflow as tf
import logging
from tqdm import tqdm

from ..coco.make_dataset import COCODataLoader, coco_names, coco_colors

logger = logging.getLogger(__name__)


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

        # # w,h->x_max,y_max
        # scale = np.array([w, h, w, h])
        # target[:, 3:] += target[:, 1:3]
        # target[:, 1:] /= scale

        return image, target

    def _load(self):
        def data_iter(source):
            for batch_data in source:
                yield self._transfer(batch_data)

        set_train = tf.data.Dataset.from_generator(generator=lambda: data_iter(self._data_source_train),
                                                   output_signature=(
                                                       tf.TensorSpec(shape=self._train_resize + (3,), dtype=tf.float32),
                                                       tf.TensorSpec(shape=(None, 5), dtype=tf.float32)
                                                   )).shuffle(self._shuffle_buffer)

        set_val = tf.data.Dataset.from_generator(generator=lambda: data_iter(self._data_source_val),
                                                 output_signature=(
                                                     tf.TensorSpec(shape=self._train_resize + (3,), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, 5), dtype=tf.float32)
                                                 )).shuffle(self._shuffle_buffer)

        return set_train, set_val

    def get_dataset(self):
        return self._train_set, self._val_set

    def draw_bbox(self, batch_data):
        image, target = batch_data
        image = image.numpy()
        target = target.numpy()
        if len(np.shape(image)) > 3:
            image = image[0]
            target = target[0]

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w = self._train_resize

        for cls, x_min, y_min, x_max, y_max in target:
            cv2.rectangle(image, (int(x_min * w), int(y_min * h)), (int(x_max * w), int(y_max * h)), (255, 255, 255))
            cv2.putText(image, coco_names[int(cls)], (int(y_min), int(x_min)), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        coco_colors[int(cls)], 2)

        return image


if __name__ == '__main__':
    loader = SSDDataset("../../datasets/coco",
                        shuffle_buffer=10,
                        batch_size=10)

    for my_image, my_target in tqdm(loader.get_dataset()[1]):
        # print(my_image, my_target)
        # cv2.imshow("preview", loader.draw_bbox((my_image, my_target)))
        # cv2.waitKey(1)
        pass
