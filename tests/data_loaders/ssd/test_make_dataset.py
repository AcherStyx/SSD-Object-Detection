from data_loaders.ssd.make_dataset import *

import unittest
import tqdm
import numpy


class TestSSDDataset(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self.dataset = SSDDataset("datasets/coco")
        except ValueError:
            self.dataset = SSDDataset("../../../datasets/coco")

        self.train, self.val = self.dataset.get_dataset()
