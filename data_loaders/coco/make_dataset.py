import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

from pycocotools.coco import COCO
from skimage import io

from templates import *


class COCODataLoader(DataLoaderTemplate):
    def load(self, *args):
        self.config: COCODataLoaderConfig

        def gen():
            coco_instance = COCO(self.config.ANNOTATION_FILE_PATH)
            img_ids = coco_instance.getImgIds()
            img_info = coco_instance.loadImgs(img_ids)
            for img_desc in img_info:
                image_raw = io.imread(img_desc['coco_url']) / 255

                annotation_ids = coco_instance.getAnnIds(img_desc['id'])
                annotations = coco_instance.loadAnns(annotation_ids)
                bbox = np.array([[x['category_id']] + x['bbox'] for x in annotations])

                yield image_raw, bbox

        self.dataset = tf.data.Dataset.from_generator(gen,
                                                      output_types=(tf.float32, tf.float32))
        self.dataset.prefetch(1000)

    @staticmethod
    def draw_bbox(image, label):
        print(image, label)
        image_numpy = image.numpy()
        for cat, x, y, w, h in label:
            cv2.rectangle(image_numpy, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (255, 255, 255), 1)

        return image_numpy


class COCODataLoaderConfig(ConfigTemplate):
    def __init__(self, annotation_file_path):
        self.ANNOTATION_FILE_PATH = annotation_file_path


# test case
if __name__ == '__main__':
    annotation_file = "dataset/coco/annotations/instances_val2017.json"
    my_config = COCODataLoaderConfig(annotation_file)
    my_set = COCODataLoader(config=my_config)

    for img, label in my_set.dataset:
        plt.imshow(my_set.draw_bbox(img, label))
        plt.show()
