__all__ = ["COCODataLoader"]

import tensorflow as tf
import cv2
import numpy as np
import os
import logging

from pycocotools.coco import COCO
from skimage import io

logger = logging.getLogger(__name__)


class COCODataLoader:
    def __init__(self, dataset_root, shuffle_buffer,
                 prefetch=10):

        self._VALID_ANNOTATION = os.path.join(dataset_root, "annotations", "instances_val2017.json")
        self._VALID_IMAGE = os.path.join(dataset_root, "val2017")
        self._TRAIN_ANNOTATION = os.path.join(dataset_root, "annotations", "instances_train2017.json")
        self._TRAIN_IMAGE = os.path.join(dataset_root, "train2017")
        self._SHUFFLE_BUFFER = shuffle_buffer
        self._PREFETCH = prefetch

        def check(file):
            if not os.path.exists(file):
                logger.critical("COCO dataset file not exist: {}".format(file))
                raise ValueError

        check(self._VALID_ANNOTATION)
        check(self._TRAIN_ANNOTATION)
        try:
            check(self._VALID_IMAGE)
        except ValueError:
            logger.warning("Pictures will from testing set be downloaded from the network in real time, "
                           "and the execution speed of the program is likely to slow down due to this.")
            self._VALID_IMAGE = None
        try:
            check(self._TRAIN_IMAGE)
        except ValueError:
            logger.warning("Pictures from training set will be downloaded from the network in real time, "
                           "and the execution speed of the program is likely to slow down due to this.")
            self._TRAIN_IMAGE = None

        super(COCODataLoader, self).__init__()
        self._train_set, self._val_set = self.load()

    @staticmethod
    def gen(annotation, image_root):
        """
        load image and label with python generator
        @param annotation: path to annotation file
        @param image_root: path to image root, None means download from the `coco_url` in annotation file
        """
        coco_instance = COCO(annotation)
        img_ids = coco_instance.getImgIds()
        img_info = coco_instance.loadImgs(img_ids)
        for img_desc in img_info:
            if image_root is not None:
                image_raw = io.imread(os.path.join(image_root, img_desc['file_name'])) / 255
            else:
                logger.debug("Downloading image: {}".format(img_desc['coco_url']))
                image_raw = io.imread(img_desc['coco_url']) / 255

            annotation_ids = coco_instance.getAnnIds(img_desc['id'])
            annotations = coco_instance.loadAnns(annotation_ids)
            bbox = np.array([[x['category_id'] - 1] + x['bbox'] for x in annotations])

            yield image_raw, bbox

    def load(self):
        train_set = tf.data.Dataset.from_generator(lambda: self.gen(self._TRAIN_ANNOTATION, self._TRAIN_IMAGE),
                                                   output_types=(tf.float32, tf.float32)
                                                   ).shuffle(self._SHUFFLE_BUFFER).prefetch(self._PREFETCH)
        valid_set = tf.data.Dataset.from_generator(lambda: self.gen(self._VALID_ANNOTATION, self._VALID_IMAGE),
                                                   output_types=(tf.float32, tf.float32)
                                                   ).shuffle(self._SHUFFLE_BUFFER).prefetch(self._PREFETCH)

        return train_set, valid_set

    @staticmethod
    def draw_bbox(image, label):
        # print(image, label)
        image_numpy = image.numpy()
        for cat, x, y, w, h in label:
            cv2.rectangle(image_numpy, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (255, 255, 255), 1)

        return image_numpy

    def get_dataset(self):
        return self._train_set, self._val_set


# test case
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    my_set = COCODataLoader(dataset_root="../../datasets/coco/", shuffle_buffer=1)

    for img, label_ in my_set.get_dataset()[1]:
        cv2.imshow("Press any key to continue", my_set.draw_bbox(img, label_))
        print(label_.numpy())
        cv2.waitKey(0)
