import unittest

from data_loaders.ssd import SSDDataset
from models.ssd_model import *


class TestSSDObjectDetectionModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.model = SSDObjectDetectionModel(classes=80)
        self.model.show_summary()

        self.dummy_input = tf.random.normal([5, 300, 300, 3])
        self.dummy_output = self.model.get_tf_model()(self.dummy_input)

        try:
            self.coco_dataset_train, self.coco_dataset_val = SSDDataset("./datasets/coco").get_dataset()
        except ValueError:
            self.coco_dataset_train, self.coco_dataset_val = SSDDataset("../../datasets/coco").get_dataset()

        super().__init__(*args, **kwargs)

    def test_train_set_visualize(self):
        dataset_for_train = self.model.get_train_set(self.coco_dataset_train)
        count = 0
        for image, (cls, bbox, mask) in dataset_for_train:
            result_image = self.model.visualize_dataset(image, cls, bbox, mask)
            cv2.imshow("view", result_image)
            cv2.waitKey(1)
            count += 1
            if count > 100:
                break

    def test_prior_box(self):
        prior_box = self.model.get_prior_box()

        print("==========0-100 prior box==========")
        for i, box in enumerate(prior_box):
            print(box)
            if i > 100:
                break
        print("==========0-100 prior box==========")


if __name__ == '__main__':
    unittest.main()
