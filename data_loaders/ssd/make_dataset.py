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
                 shuffle_buffer=1,
                 train_size=(300, 300),
                 dataset="COCO"):
        self._train_resize = train_size
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
        h, w, _ = image.shape
        image = cv2.resize(image, self._train_resize)

        # relative result
        scale = np.array([w, h, w, h])
        target[:, 1:3] += target[:, 3:] / 2
        target[:, 1:] /= scale

        return image, target

    def _load(self):
        def data_iter(source):
            for batch_data in source.as_numpy_iterator():
                image, target = self._transfer(batch_data)
                yield image, target

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

        h_scale, w_scale = self._train_resize

        for cls, x, y, w, h in target:
            x_min, y_min, x_max, y_max = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            cv2.rectangle(image, (int(x_min * w_scale), int(y_min * h_scale)),
                          (int(x_max * w_scale), int(y_max * h_scale)), (255, 255, 255))
            cv2.putText(image, coco_names[int(cls)], (int(x_min * h_scale), int(y_min * w_scale)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        coco_colors[int(cls)], 2)

        return image


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    loader = SSDDataset("../../datasets/coco",
                        shuffle_buffer=1)

    for my_image, my_target in tqdm(loader.get_dataset()[1]):
        print(my_target)
        cv2.imshow("preview", loader.draw_bbox((my_image, my_target)))
        cv2.waitKey(1)
        pass
