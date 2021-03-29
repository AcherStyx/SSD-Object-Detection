import logging
import itertools
import tensorflow as tf
import math
import numpy as np
from tensorflow.keras import layers, Model

from utils.bbox import match_bbox

logger = logging.getLogger(__name__)


class SSDObjectDetectionModel:
    def __init__(self,
                 classes):
        self.INPUT_SHAPE = (300, 300, 3)
        self.CLASSES = classes
        self.THRESH = 0.5

        self.FILTER_SIZE_1 = 4 * (self.CLASSES + 4)
        self.FILTER_SIZE_2 = 6 * (self.CLASSES + 4)
        self.SAMPLE_SIZE = (self.CLASSES + 4)

        self._prior_box, self._model = self._build()

    def _build(self):

        args_3_512 = {"filters": 512,
                      "kernel_size": (3, 3),
                      "padding": "SAME",
                      "activation": "relu"}
        args_1_512 = {"filters": 512,
                      "kernel_size": (1, 1),
                      "padding": "SAME",
                      "activation": "relu"}
        args_pool = {"pool_size": (2, 2), "strides": (2, 2), "padding": "SAME"}

        input_layer = layers.Input(shape=(300, 300, 3))

        model = tf.keras.applications.VGG16(include_top=False, input_shape=(300, 300, 3))
        pre_trained_vgg = Model(inputs=model.input, outputs=model.get_layer("block3_conv3").output)(input_layer)

        hidden_layer = layers.MaxPool2D(**args_pool)(pre_trained_vgg)

        hidden_layer = layers.Conv2D(**args_3_512)(hidden_layer)
        hidden_layer = layers.Conv2D(**args_3_512)(hidden_layer)
        hidden_layer = layers.Conv2D(**args_1_512)(hidden_layer)

        # ssd head
        feature_map_1 = hidden_layer

        hidden_layer = layers.Conv2D(filters=1024,
                                     kernel_size=(3, 3),
                                     strides=(2, 2),
                                     activation="relu",
                                     padding="SAME")(feature_map_1)
        feature_map_2 = layers.Conv2D(filters=1024,
                                      kernel_size=(1, 1),
                                      activation="relu",
                                      padding="SAME")(hidden_layer)

        hidden_layer = layers.Conv2D(filters=256,
                                     kernel_size=(1, 1),
                                     activation='relu',
                                     padding="SAME")(feature_map_2)
        feature_map_3 = layers.Conv2D(filters=512,
                                      kernel_size=(3, 3),
                                      strides=(2, 2),
                                      activation="relu",
                                      padding="SAME")(hidden_layer)

        hidden_layer = layers.Conv2D(filters=128,
                                     kernel_size=(1, 1),
                                     activation="relu",
                                     padding="SAME")(feature_map_3)
        feature_map_4 = layers.Conv2D(filters=256,
                                      kernel_size=(3, 3),
                                      strides=(2, 2),
                                      activation="relu",
                                      padding="SAME")(hidden_layer)

        hidden_layer = layers.Conv2D(filters=128,
                                     kernel_size=(1, 1),
                                     activation="relu",
                                     padding="SAME")(feature_map_4)
        feature_map_5 = layers.Conv2D(filters=256,
                                      kernel_size=(3, 3),
                                      activation="relu")(hidden_layer)

        hidden_layer = layers.Conv2D(filters=128,
                                     kernel_size=(1, 1),
                                     activation="relu",
                                     padding="SAME")(feature_map_5)
        feature_map_6 = layers.Conv2D(filters=256,
                                      kernel_size=(3, 3),
                                      activation="relu")(hidden_layer)

        # detection result
        detection_1 = layers.Conv2D(filters=self.FILTER_SIZE_1,
                                    kernel_size=(3, 3),
                                    activation=None,
                                    padding="SAME")(feature_map_1)
        detection_2 = layers.Conv2D(filters=self.FILTER_SIZE_2,
                                    kernel_size=(3, 3),
                                    activation=None,
                                    padding="SAME")(feature_map_2)
        detection_3 = layers.Conv2D(filters=self.FILTER_SIZE_2,
                                    kernel_size=(3, 3),
                                    activation=None,
                                    padding="SAME")(feature_map_3)
        detection_4 = layers.Conv2D(filters=self.FILTER_SIZE_2,
                                    kernel_size=(3, 3),
                                    activation=None,
                                    padding="SAME")(feature_map_4)
        detection_5 = layers.Conv2D(filters=self.FILTER_SIZE_1,
                                    kernel_size=(3, 3),
                                    activation=None,
                                    padding="SAME")(feature_map_5)
        detection_6 = layers.Conv2D(filters=self.FILTER_SIZE_1,
                                    kernel_size=(3, 3),
                                    activation=None,
                                    padding="SAME")(feature_map_6)

        # calculate prior box for train
        prior_box = []
        size_list = [detection_1.shape[1:3], detection_2.shape[1:3], detection_3.shape[1:3],
                     detection_4.shape[1:3], detection_5.shape[1:3], detection_6.shape[1:3]]
        size_list = [list(x) for x in size_list]
        s_k_refer = [21, 45, 99, 153, 207, 261, 315]
        aspect_ratio = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        for index, (h, w) in enumerate(size_list):
            for x, y in itertools.product(range(w), range(h), repeat=1):
                cx = (x + 0.5) / w
                cy = (y + 0.5) / h

                # TODO: use calculate result
                s_k = s_k_refer[index] / self.INPUT_SHAPE[0]
                prior_box.append([cx, cy, s_k, s_k])

                s_k_prime = math.sqrt(s_k * (s_k_refer[index + 1] / self.INPUT_SHAPE[0]))
                prior_box.append([cx, cy, s_k_prime, s_k_prime])

                for ratio in aspect_ratio[index]:
                    prior_box.append([cx, cy, s_k * math.sqrt(ratio), s_k / math.sqrt(ratio)])
                    prior_box.append([cx, cy, s_k / math.sqrt(ratio), s_k * math.sqrt(ratio)])

        detection_1 = layers.Reshape(target_shape=(-1, self.CLASSES + 4))(detection_1)
        detection_2 = layers.Reshape(target_shape=(-1, self.CLASSES + 4))(detection_2)
        detection_3 = layers.Reshape(target_shape=(-1, self.CLASSES + 4))(detection_3)
        detection_4 = layers.Reshape(target_shape=(-1, self.CLASSES + 4))(detection_4)
        detection_5 = layers.Reshape(target_shape=(-1, self.CLASSES + 4))(detection_5)
        detection_6 = layers.Reshape(target_shape=(-1, self.CLASSES + 4))(detection_6)
        output_layer = layers.Concatenate(axis=-2)([detection_1, detection_2, detection_3,
                                                    detection_4, detection_5, detection_6])
        loc = output_layer[..., :4]
        conf = layers.Softmax()(output_layer[..., 4:])

        assert len(prior_box) == int(conf.shape[1])

        return prior_box, Model(inputs=input_layer,
                                outputs=[loc, conf],
                                name="SSDObjectDetectionModel")

    @staticmethod
    @tf.function
    def _get_single_level_center_point(featmap_size, image_size, dtype=tf.float32, flatten=False):
        strides = image_size / featmap_size
        h, w = featmap_size

        x_range = (tf.range(w, dtype=dtype) + 0.5) * strides
        y_range = (tf.range(h, dtype=dtype) + 0.5) * strides
        y, x = tf.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return x, y

    @tf.function
    def _single_loss(self, y_true, y_pred):
        """
        calculate location and classification loss for single image
        @param y_true: [[cls, x, y, w, h], ...]
        @param y_pred: ssd featmap
        """
        loss = tf.constant(0.0, dtype=tf.float32)

        loc, conf = y_pred

        return loss

    def train(self, dataset):

        def batch_data_iter(tf_dataset: tf.data.Dataset, prior_box, thresh):
            for iter_image, iter_targets in tf_dataset.as_numpy_iterator():
                matched_cls, matched_loc, matched_mask = match_bbox(iter_targets, prior_box, thresh)

                yield iter_image, matched_cls, matched_loc, matched_mask

        dataset = tf.data.Dataset.from_generator(
            generator=lambda: batch_data_iter(dataset, self._prior_box, self.THRESH),
            output_signature=(
                tf.TensorSpec(self.INPUT_SHAPE, dtype=tf.float32),
                tf.TensorSpec((np.shape(self._prior_box)[0],), dtype=tf.int32),
                tf.TensorSpec(np.shape(self._prior_box), dtype=tf.float32),
                tf.TensorSpec((np.shape(self._prior_box)[0],), dtype=tf.bool)
            )
        ).batch(2)

        for a, b, c, d in dataset:
            print(a, b, c, d)

    def show_summary(self):
        self._model.summary()
        tf.keras.utils.plot_model(self._model,
                                  to_file=str(self.__class__.__name__) + ".png",
                                  show_shapes=True,
                                  dpi=50)

    def get_tf_model(self):
        return self._model

    def get_prior_box(self):
        return self._prior_box


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    my_model = SSDObjectDetectionModel(classes=80)
    my_model.show_summary()
    dummy_input = tf.random.normal([5, 300, 300, 3])
    dummy_output = my_model.get_tf_model()(dummy_input)
    for i, out in enumerate(dummy_output):
        print("output {}: {}".format(i, out.shape))
    print(np.shape(my_model.get_prior_box()))

    from data_loaders.coco import COCODataLoader

    data = COCODataLoader("../datasets/coco").get_dataset()[0]
    my_model.train(data)
