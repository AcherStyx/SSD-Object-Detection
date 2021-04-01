import unittest

import cv2
from tqdm import tqdm

from data_loaders.coco.make_dataset import *


class TestCOCODataLoader(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self.dataset = COCODataLoader("datasets/coco")
        except ValueError:
            self.dataset = COCODataLoader("../../../datasets/coco")

        self.dataset_train, self.dataset_val = self.dataset.get_dataset()

    def test_visualize(self):
        print("visualize...")
        count = 0
        for image, cls, bbox in self.dataset_train:
            image = self.dataset.draw_bbox(image, cls, bbox)
            cv2.imshow("test_view", image)
            cv2.waitKey(1)
            count += 1
            if count > 100:
                break
        cv2.destroyWindow("test_view")
        print("done")

    def test_data(self):
        print("test load all data...")
        for _ in tqdm(self.dataset_train):
            pass
        print("done")


if __name__ == '__main__':
    unittest.main()
