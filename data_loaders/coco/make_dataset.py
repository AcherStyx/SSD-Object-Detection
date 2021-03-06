__all__ = ["COCODataLoader", "coco_names", "coco_colors"]

import tensorflow as tf
import cv2
import numpy as np
import os
import logging
import pickle

from pycocotools.coco import COCO
from skimage import io
from tqdm import tqdm

from utils.bbox import draw_bbox

logger = logging.getLogger(__name__)

coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

coco_colors = [np.random.randint(80, 240, (3,), dtype=np.uint8).tolist() for _ in range(len(coco_names))]


class COCODataLoader:
    def __init__(self, dataset_root,
                 prefetch=1,
                 shuffle=True,
                 mini_batch=0):
        """
        load object detection data from coco dataset
        @param dataset_root: path to the dataset
        @param prefetch:
        @param shuffle: use shuffle
        """
        self._PREFETCH = prefetch
        self._SHUFFLE = shuffle

        self._VAL_ANNOTATION = os.path.join(dataset_root, "annotations", "instances_val2017.json")
        self._VAL_IMAGE = os.path.join(dataset_root, "val2017")
        self._TRAIN_ANNOTATION = os.path.join(dataset_root, "annotations", "instances_train2017.json")
        self._TRAIN_IMAGE = os.path.join(dataset_root, "train2017")
        self._MINI_BATCH = mini_batch

        def check(file):
            if not os.path.exists(file):
                logger.critical("COCO dataset file not exist: {}".format(file))
                raise ValueError

        check(self._VAL_ANNOTATION)
        check(self._TRAIN_ANNOTATION)
        try:
            check(self._VAL_IMAGE)
        except ValueError:
            logger.warning("Pictures will from testing set be downloaded from the network in real time, "
                           "and the execution speed of the program is likely to slow down due to this.")
            self._VAL_IMAGE = None
        try:
            check(self._TRAIN_IMAGE)
        except ValueError:
            logger.warning("Pictures from training set will be downloaded from the network in real time, "
                           "and the execution speed of the program is likely to slow down due to this.")
            self._TRAIN_IMAGE = None

        train_cache_file = os.path.join(dataset_root, ".annotation_cache_train")
        val_cache_file = os.path.join(dataset_root, ".annotation_cache_val")

        try:
            with open(train_cache_file, "rb") as f:
                self._coco_instance_train = pickle.load(f)
            with open(val_cache_file, "rb") as f:
                self._coco_instance_val = pickle.load(f)
        except FileNotFoundError:
            self._coco_instance_train = COCO(self._TRAIN_ANNOTATION)
            self._coco_instance_val = COCO(self._VAL_ANNOTATION)
            with open(train_cache_file, "wb") as f:
                pickle.dump(self._coco_instance_train, f)
            with open(val_cache_file, "wb") as f:
                pickle.dump(self._coco_instance_val, f)

        self._label_transfer_dict = self._load_label_transfer_dict()
        self._train_set, self._val_set = self.load()

    def _load_label_transfer_dict(self):
        assert self._coco_instance_train.cats == self._coco_instance_val.cats

        refer_dict = {}
        for index, label_name, coco_id_info in zip(range(80), coco_names, self._coco_instance_train.cats.items()):
            assert label_name == coco_id_info[1]["name"]
            refer_dict[coco_id_info[0]] = index
        return refer_dict

    def gen(self, coco_instance, image_root):
        """
        load image and label with python generator
        @param coco_instance: path to annotation file
        @param image_root: path to image root, None means download from the `coco_url` in annotation file
        """
        img_ids = coco_instance.getImgIds()
        # TODO: remove debug setting
        if self._MINI_BATCH:
            img_info = coco_instance.loadImgs(img_ids)[:int(self._MINI_BATCH)]
        else:
            img_info = coco_instance.loadImgs(img_ids)
        if self._SHUFFLE:
            np.random.shuffle(img_info)

        for img_desc in img_info:
            if image_root is not None:
                image_raw = io.imread(os.path.join(image_root, img_desc['file_name'])) / 255
            else:
                logger.debug("Downloading image: {}".format(img_desc['coco_url']))
                image_raw = io.imread(img_desc['coco_url']) / 255

            annotation_ids = coco_instance.getAnnIds(img_desc['id'])
            annotations = coco_instance.loadAnns(annotation_ids)
            bbox = np.array([x['bbox'] for x in annotations])
            cls = np.array([self._label_transfer_dict[x['category_id']] for x in annotations])
            if len(np.shape(bbox)) < 2:
                logger.debug("No object in this image")
                continue
            if len(np.shape(image_raw)) == 2:
                image_raw = np.stack([image_raw, image_raw, image_raw], axis=2)

            bbox[:, :2] += bbox[:, 2:] / 2

            yield image_raw, cls, bbox

    def load(self):
        train_set = tf.data.Dataset.from_generator(lambda: self.gen(self._coco_instance_train, self._TRAIN_IMAGE),
                                                   output_signature=(tf.TensorSpec((None, None, 3), dtype=tf.float32),
                                                                     tf.TensorSpec((None,), dtype=tf.float32),
                                                                     tf.TensorSpec((None, 4), dtype=tf.float32))
                                                   ).prefetch(self._PREFETCH)
        valid_set = tf.data.Dataset.from_generator(lambda: self.gen(self._coco_instance_val, self._VAL_IMAGE),
                                                   output_signature=(tf.TensorSpec((None, None, 3), dtype=tf.float32),
                                                                     tf.TensorSpec((None,), dtype=tf.float32),
                                                                     tf.TensorSpec((None, 4), dtype=tf.float32))
                                                   ).prefetch(self._PREFETCH)

        return train_set, valid_set

    @staticmethod
    def draw_bbox(image, cls, bbox):
        image = image.numpy()
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return draw_bbox(image,
                         bbox=bbox.numpy(),
                         cls_label=cls.numpy(),
                         cls_names=coco_names,
                         cls_color=coco_colors)

    def get_dataset(self):
        return self._train_set, self._val_set


# test case
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    my_set = COCODataLoader(dataset_root="../../datasets/coco/", prefetch=10)

    for img_, cls_, bbox_ in tqdm(my_set.get_dataset()[1]):
        cv2.imshow("Press any key to continue", my_set.draw_bbox(img_, cls_, bbox_))
        cv2.waitKey(0)
