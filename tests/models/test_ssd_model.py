import unittest
from tensorflow.keras import optimizers

from data_loaders.ssd import SSDDataLoader
from models.ssd_model import *


class TestSSDObjectDetectionModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        try:
            self.dataset = SSDDataLoader("./datasets/coco")
            self.coco_dataset_train, self.coco_dataset_val = self.dataset.get_dataset()
            self.model = SSDObjectDetectionModel(classes=80, log_dir="./workshop/ssd")
            self.model.show_summary()
        except ValueError:
            self.dataset = SSDDataLoader("../../datasets/coco")
            self.coco_dataset_train, self.coco_dataset_val = self.dataset.get_dataset()
            self.model = SSDObjectDetectionModel(classes=80, log_dir="../../workshop/ssd")
            self.model.show_summary()

        self.dummy_input = tf.random.normal([5, 300, 300, 3])
        self.dummy_output = self.model.get_tf_model()(self.dummy_input)

        super().__init__(*args, **kwargs)

    def test_train_set_visualize(self):
        dataset_for_train = self.model.get_train_set(self.coco_dataset_train)
        for image, (cls, bbox, mask) in dataset_for_train:
            result_image = self.model.visualize_dataset(image, cls, bbox, mask)
            cv2.imshow("view", result_image)
            ch = cv2.waitKey(0)
            if ch == "q":
                break

    def test_prior_box(self):
        prior_box = self.model.get_prior_box()

        print("==========prior box==========")
        for i, box in enumerate(prior_box):
            print(i, box)
            # if i > 1000:
            #     break
        print("==========prior box==========")

    def test_prior_box_visualize(self):
        self.model.visualize_prior_box()

    def test_train(self):
        logging.basicConfig(level=logging.INFO)
        # self.model.load()
        lr_schedule = optimizers.schedules.ExponentialDecay(0.0005, 500, 0.999)
        self.model.train(self.dataset, epoch=100, batch_size=8,
                         optimizer=optimizers.Adam(lr_schedule),
                         warmup=True, visualization_log_interval=500, warmup_step=2000,
                         warmup_optimizer=optimizers.Adam(optimizers.schedules.PolynomialDecay(1e-6, 2000, 1e-3)))
        self.model.save()


if __name__ == '__main__':
    unittest.main()
