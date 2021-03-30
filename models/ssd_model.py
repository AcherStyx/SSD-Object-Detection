import logging
import itertools

import cv2
import tensorflow as tf
import math
import numpy as np
from tensorflow.keras import layers, Model, optimizers
from tqdm import tqdm

from utils.bbox import match_bbox, apply_anchor_box

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
        self._optimizer = optimizers.Adam()

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
        pre_trained_vgg = Model(inputs=model.input,
                                outputs=model.get_layer("block3_conv3").output
                                )(input_layer)

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
                                    activation="sigmoid",
                                    padding="SAME")(feature_map_1)
        detection_2 = layers.Conv2D(filters=self.FILTER_SIZE_2,
                                    kernel_size=(3, 3),
                                    activation="sigmoid",
                                    padding="SAME")(feature_map_2)
        detection_3 = layers.Conv2D(filters=self.FILTER_SIZE_2,
                                    kernel_size=(3, 3),
                                    activation="sigmoid",
                                    padding="SAME")(feature_map_3)
        detection_4 = layers.Conv2D(filters=self.FILTER_SIZE_2,
                                    kernel_size=(3, 3),
                                    activation="sigmoid",
                                    padding="SAME")(feature_map_4)
        detection_5 = layers.Conv2D(filters=self.FILTER_SIZE_1,
                                    kernel_size=(3, 3),
                                    activation="sigmoid",
                                    padding="SAME")(feature_map_5)
        detection_6 = layers.Conv2D(filters=self.FILTER_SIZE_1,
                                    kernel_size=(3, 3),
                                    activation="sigmoid",
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
        conf = output_layer[..., 4:]

        assert len(prior_box) == int(conf.shape[1])

        return np.array(prior_box), Model(inputs=input_layer,
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

    def train(self, dataset, epoch=1):

        def batch_data_iter(tf_dataset: tf.data.Dataset, prior_box, thresh):
            for iter_image, iter_targets in tf_dataset.as_numpy_iterator():
                matched_cls, matched_loc, matched_mask = match_bbox(iter_targets, prior_box, thresh)
                # print("---------")
                # print(matched_cls, matched_loc, self._prior_box)
                matched_loc = apply_anchor_box(matched_loc, self._prior_box)
                # print("transfer shape:", np.shape(matched_loc))
                iter_image = (iter_image - 0.5) * 2
                yield iter_image, (matched_cls, matched_loc, matched_mask)

        batch_dataset = tf.data.Dataset.from_generator(
            generator=lambda: batch_data_iter(dataset, self._prior_box, self.THRESH),
            output_signature=(
                tf.TensorSpec(self.INPUT_SHAPE, dtype=tf.float32),
                (tf.TensorSpec((np.shape(self._prior_box)[0],), dtype=tf.int32),
                 tf.TensorSpec(np.shape(self._prior_box), dtype=tf.float32),
                 tf.TensorSpec((np.shape(self._prior_box)[0],), dtype=tf.bool))
            )
        ).batch(2)

        # for image, (ground_truth_cls, ground_truth_box, mask) in tqdm(batch_dataset):
        #     self.visualize_dataset(image, ground_truth_cls, ground_truth_box, mask)

        for i in range(epoch):
            logger.info("Epoch %s/%s", i, epoch)
            for image, (ground_truth_cls, ground_truth_box, mask) in batch_dataset:
                with tf.GradientTape() as tape:
                    # print("==========train step start==========")
                    pred_loc, pred_conf = self._model(image, training=True)
                    total_loss = self._ssd_loss((ground_truth_cls, ground_truth_box, mask), (pred_loc, pred_conf))
                    # print("==========train step end==========")

                ssd_gradient = tape.gradient(total_loss, self._model.trainable_variables)
                self._optimizer.apply_gradients(
                    zip(ssd_gradient, self._model.trainable_variables)
                )
                self.visualize(image, pred_conf, pred_loc, name="predict")
                self.visualize_dataset(image, ground_truth_cls, ground_truth_box, mask, name="ground truth")

                logger.debug("SSD Training loss: %s", total_loss.numpy())

    @staticmethod
    def _ssd_loss(y_true, y_pred):
        # print("loss input: ", y_true, y_pred)
        gt_cls, gt_box, gt_mask = y_true
        pred_box, pred_cls = y_pred

        # print(tf.boolean_mask(gt_cls, mask=gt_mask), tf.boolean_mask(pred_cls, mask=gt_mask))
        # loss_cls = tf.keras.losses.sparse_categorical_crossentropy(gt_cls, pred_cls)
        # print(loss_cls)
        loss_cls = tf.reduce_sum(
            tf.keras.losses.sparse_categorical_crossentropy(tf.boolean_mask(gt_cls, gt_mask),
                                                            tf.boolean_mask(pred_cls, gt_mask))
        )
        loss_cls += tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(0.0,
                                                tf.boolean_mask(
                                                    pred_cls, tf.equal(gt_mask, tf.zeros_like(gt_mask, dtype=tf.bool)))
                                                )
        )
        print("cls loss: ", loss_cls)

        # TODO: add box loss
        loss_box = tf.reduce_sum(
            tf.keras.losses.mean_absolute_error(tf.boolean_mask(gt_box, gt_mask),
                                                tf.boolean_mask(pred_box, gt_mask))
        )
        # print("loc loss: ", loss_box)

        logger.debug("Loss function: loc: %s | cls: %s", loss_box, loss_cls)

        return loss_cls + loss_box

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

    def visualize_dataset(self, image, gt_cls, gt_bbox, mask, name="ssd visualize"):
        image = np.array(image.numpy()) / 2 + 0.5
        gt_bbox, gt_cls = np.array(gt_bbox.numpy()), np.array(gt_cls.numpy())
        mask = np.array(mask)

        if len(image.shape) > 3:
            image = image[0]
            gt_cls = gt_cls[0]
            gt_bbox = gt_bbox[0]
            mask = mask[0]

        gt_bbox_masked = gt_bbox[mask]
        gt_cls_masked = gt_cls[mask]
        default_box_masked = self._prior_box[mask]

        for cls, bbox, default_box in zip(gt_cls_masked, gt_bbox_masked, default_box_masked):
            # print(cls, bbox)
            cx, cy = (bbox[:2] * default_box[2:] + default_box[:2] + np.random.normal(scale=0.01, size=(2,))) * 300
            w, h = np.exp(bbox[2:]) * default_box[2:] * 300
            # print(image, cx, cy, w, h)
            cv2.rectangle(image, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)),
                          (1.0, 1.0, 1.0))

        cv2.imshow(name, image)
        cv2.waitKey(1)

    def visualize(self, image, pred_conf, pred_bbox, thresh=0.5, name="ssd visualize"):
        mask = tf.reduce_max(pred_conf, axis=-1) > thresh
        print(tf.reduce_max(pred_conf, axis=-1))
        print(mask)
        pred_cls = tf.argmax(pred_conf, axis=-1)
        self.visualize_dataset(image, pred_cls, pred_bbox, mask, name=name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    my_model = SSDObjectDetectionModel(classes=80)
    my_model.show_summary()
    dummy_input = tf.random.normal([5, 300, 300, 3])
    dummy_output = my_model.get_tf_model()(dummy_input)
    for i, out in enumerate(dummy_output):
        print("output {}: {}".format(i, out.shape))
    print(np.shape(my_model.get_prior_box()))

    from data_loaders.ssd import SSDDataset

    data = SSDDataset("../datasets/coco").get_dataset()[0]
    my_model.train(data)
