import logging
import itertools
import os
import time
import math

import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, optimizers, activations
from tqdm import tqdm

from utils.bbox import match_bbox, apply_anchor_box
from data_loaders.ssd import SSDDataLoader

logger = logging.getLogger(__name__)


class SSDObjectDetectionModel:
    def __init__(self,
                 classes,
                 log_dir=None,
                 learning_rate=0.001):
        self._INPUT_SHAPE = (300, 300, 3)
        self._CLASSES = classes + 1
        self._THRESH = 0.5

        self._FILTER_SIZE_1 = 4 * (self._CLASSES + 4)
        self._FILTER_SIZE_2 = 6 * (self._CLASSES + 4)
        self._SAMPLE_SIZE = (self._CLASSES + 4)

        self._prior_box, self._model = self._build()

        self._optimizer = optimizers.Adam(learning_rate=learning_rate)
        if log_dir is not None:
            time_stamp = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
            log_dir = os.path.join(log_dir, time_stamp)
            self._tensorboard_writer = tf.summary.create_file_writer(log_dir)

            # write graph
            @tf.function
            def trace_model(data, model):
                model(data)

            tf.summary.trace_on(graph=True)
            trace_model(tf.zeros((1, 300, 300, 3)), self._model)
            with self._tensorboard_writer.as_default():
                tf.summary.trace_export("SSD Model", step=0)
        else:
            self._tensorboard_writer = None  # not use tensorboard

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
                                outputs=model.get_layer("block3_conv3").output,
                                trainable=False,
                                name="pre-trained-vgg"
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
                                     activation="relu",
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
        detection_1 = layers.Conv2D(filters=self._FILTER_SIZE_1,
                                    kernel_size=(3, 3),
                                    activation=None,
                                    padding="SAME")(feature_map_1)
        detection_2 = layers.Conv2D(filters=self._FILTER_SIZE_2,
                                    kernel_size=(3, 3),
                                    activation=None,
                                    padding="SAME")(feature_map_2)
        detection_3 = layers.Conv2D(filters=self._FILTER_SIZE_2,
                                    kernel_size=(3, 3),
                                    activation=None,
                                    padding="SAME")(feature_map_3)
        detection_4 = layers.Conv2D(filters=self._FILTER_SIZE_2,
                                    kernel_size=(3, 3),
                                    activation=None,
                                    padding="SAME")(feature_map_4)
        detection_5 = layers.Conv2D(filters=self._FILTER_SIZE_1,
                                    kernel_size=(3, 3),
                                    activation=None,
                                    padding="SAME")(feature_map_5)
        detection_6 = layers.Conv2D(filters=self._FILTER_SIZE_1,
                                    kernel_size=(3, 3),
                                    activation=None,
                                    padding="SAME")(feature_map_6)

        prior_box = self._build_prior_box(
            size_list=[detection_1.shape[1:3], detection_2.shape[1:3], detection_3.shape[1:3],
                       detection_4.shape[1:3], detection_5.shape[1:3], detection_6.shape[1:3]]
        )

        detection_1 = layers.Reshape(target_shape=(-1, self._CLASSES + 4))(detection_1)
        detection_2 = layers.Reshape(target_shape=(-1, self._CLASSES + 4))(detection_2)
        detection_3 = layers.Reshape(target_shape=(-1, self._CLASSES + 4))(detection_3)
        detection_4 = layers.Reshape(target_shape=(-1, self._CLASSES + 4))(detection_4)
        detection_5 = layers.Reshape(target_shape=(-1, self._CLASSES + 4))(detection_5)
        detection_6 = layers.Reshape(target_shape=(-1, self._CLASSES + 4))(detection_6)
        output_layer = layers.Concatenate(axis=-2)([detection_1, detection_2, detection_3,
                                                    detection_4, detection_5, detection_6])

        loc = output_layer[..., :4]
        conf = output_layer[..., 4:]

        return prior_box, Model(inputs=input_layer,
                                outputs=[loc, conf],
                                name="SSDObjectDetectionModel")

    def _build_prior_box(self, size_list):
        prior_box = []
        size_list = [list(x) for x in size_list]
        s_k_refer = [21, 45, 99, 153, 207, 261, 315]
        aspect_ratio = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        for index, (h, w) in enumerate(size_list):
            for y, x in itertools.product(range(h), range(w), repeat=1):
                cx = (x + 0.5) / w
                cy = (y + 0.5) / h

                # TODO: use calculate result
                s_k = s_k_refer[index] / self._INPUT_SHAPE[0]
                prior_box.append([cx, cy, s_k, s_k])

                s_k_prime = math.sqrt(s_k * (s_k_refer[index + 1] / self._INPUT_SHAPE[0]))
                prior_box.append([cx, cy, s_k_prime, s_k_prime])

                for ratio in aspect_ratio[index]:
                    prior_box.append([cx, cy, s_k * math.sqrt(ratio), s_k / math.sqrt(ratio)])
                    prior_box.append([cx, cy, s_k / math.sqrt(ratio), s_k * math.sqrt(ratio)])

        return np.array(prior_box)

    @staticmethod
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

    def get_train_set(self, dataset, batch_size=1):
        def batch_data_iter(tf_dataset: tf.data.Dataset, prior_box, thresh):
            for iter_image, iter_cls, iter_bbox in tf_dataset.as_numpy_iterator():
                matched_cls, matched_loc, matched_mask = match_bbox(iter_cls, iter_bbox, prior_box, thresh)
                matched_loc = apply_anchor_box(matched_loc, self._prior_box)
                iter_image = (iter_image - 0.5) * 2
                yield iter_image, (matched_cls, matched_loc, matched_mask)

        batch_dataset = tf.data.Dataset.from_generator(
            generator=lambda: batch_data_iter(dataset, self._prior_box, self._THRESH),
            output_signature=(
                tf.TensorSpec(self._INPUT_SHAPE, dtype=tf.float32),
                (tf.TensorSpec((np.shape(self._prior_box)[0],), dtype=tf.int32),
                 tf.TensorSpec(np.shape(self._prior_box), dtype=tf.float32),
                 tf.TensorSpec((np.shape(self._prior_box)[0],), dtype=tf.bool))
            )
        ).batch(batch_size, drop_remainder=True).prefetch(10)

        return batch_dataset

    def _train_step(self, image, gt_cls, gt_bbox, gt_mask, ssd_optimizer, stage, set_names, set_colors, step):
        with tf.GradientTape() as tape:
            pred_loc, pred_conf = self._model(image, training=True)
            total_loss, info = self._ssd_loss((gt_cls, gt_bbox, gt_mask), (pred_loc, pred_conf))
        ssd_gradient = tape.gradient(total_loss, self._model.trainable_variables)
        ssd_gradient = [tf.clip_by_norm(x, 0.01) for x in ssd_gradient]
        ssd_optimizer.apply_gradients(
            zip(ssd_gradient, self._model.trainable_variables)
        )

        info["lr"] = ssd_optimizer.lr.numpy()
        if step % 10 == 0:
            img_ssd_pred = self.visualize(image, pred_conf, pred_loc,
                                          name="ssd_pred", thresh=0.3, show=False,
                                          label_names=set_names, label_colors=set_colors)
            img_pred_with_mask = self.visualize(image, pred_conf, pred_loc,
                                                name="ssd_pred_with_mask", thresh=0.3, show=False,
                                                mask=gt_mask,
                                                label_names=set_names, label_colors=set_colors)
            img_ssd_gt = self.visualize_dataset(image, gt_cls, gt_bbox, gt_mask,
                                                name="ssd_gt", show=False,
                                                label_names=set_names, label_colors=set_colors)
            tf.summary.image(stage + "/pred", np.expand_dims(img_ssd_pred, axis=0), step=step)
            tf.summary.image(stage + "/pred_with_mask", np.expand_dims(img_pred_with_mask, axis=0), step=step)
            tf.summary.image(stage + "/gt", np.expand_dims(img_ssd_gt, axis=0), step=step)

        tf.summary.scalar(stage + "/loc loss", info["loc loss"], step=step)
        tf.summary.scalar(stage + "/cls loss pos", info["cls loss pos"], step=step)
        tf.summary.scalar(stage + "/cls loss neg", info["cls loss neg"], step=step)
        tf.summary.scalar(stage + "/lr", info["lr"], step=step)

        return pred_conf, pred_loc, info

    def _train(self, data_loader: SSDDataLoader,
               epoch, batch_size, optimizer,
               warmup, warmup_step, warmup_lr):

        train_set, val_set = data_loader.get_dataset()
        set_names, set_colors = data_loader.get_names_and_colors()
        batch_dataset = self.get_train_set(train_set, batch_size)

        if optimizer is not None:
            self._optimizer = optimizer
            self._optimizer: tf.optimizers.Optimizer

        if warmup:
            logger.info("Warm up for %s steps, lr %s -> %s", warmup_step, warmup_lr[0], warmup_lr[1])
            step = 0
            warmup_optimizer = optimizers.SGD(momentum=0.9)
            bar = tqdm(total=warmup_step)
            while True:
                for image, (gt_cls, gt_bbox, gt_mask) in batch_dataset:
                    bar.update(1)
                    step += 1
                    learning_rate = step * (warmup_lr[1] - warmup_lr[0]) / warmup_step + warmup_lr[0]
                    warmup_optimizer.lr = learning_rate
                    logger.debug("Warm up with learning rate %s", learning_rate)
                    pred_conf, pred_loc, info = self._train_step(image, gt_cls, gt_bbox, gt_mask, warmup_optimizer,
                                                                 "warmup", set_names, set_colors, step)
                    bar.set_postfix(info)

                    if step >= warmup_step:
                        break
                if step >= warmup_step:
                    bar.close()
                    break

        step = 0
        for i in range(epoch):
            logger.info("Epoch %s/%s", i + 1, epoch)
            bar = tqdm()
            for image, (gt_cls, gt_bbox, gt_mask) in batch_dataset:
                step += 1
                pred_conf, pred_loc, info = self._train_step(image, gt_cls, gt_bbox, gt_mask, self._optimizer,
                                                             "train", set_names, set_colors, step)
                bar.update(1)
                bar.set_postfix(info)
            bar.close()

    def train(self, data_loader: SSDDataLoader,
              epoch=1, batch_size=1, optimizer=None,
              warmup=True, warmup_step=1000, warmup_lr=(0.000001, 0.001)):
        if self._tensorboard_writer is not None:
            with self._tensorboard_writer.as_default():
                self._train(data_loader, epoch, batch_size, optimizer, warmup, warmup_step, warmup_lr)
        else:
            self._train(data_loader, epoch, batch_size, optimizer, warmup, warmup_step, warmup_lr)

    @staticmethod
    def _ssd_loss(y_true, y_pred):
        gt_cls, gt_box, gt_mask = y_true
        pred_box, pred_cls = y_pred

        pred_box = tf.tanh(pred_box)

        # have same batch size
        assert tf.shape(gt_cls)[0] == tf.shape(gt_box)[0] == tf.shape(gt_mask)[0] == tf.shape(pred_box)[0] == \
               tf.shape(pred_cls)[0]
        # number of prior box == number of output box
        assert tf.boolean_mask(pred_cls, tf.equal(gt_mask, tf.zeros_like(gt_mask, dtype=tf.bool))).shape[0] + \
               tf.boolean_mask(gt_cls, gt_mask).shape[0] == gt_cls.shape[0] * gt_cls.shape[1]

        batch_size = tf.cast(tf.shape(gt_mask)[0], tf.float32)

        # cls loss positive
        bool_positive_mask = gt_mask
        float_positive_mask = tf.cast(bool_positive_mask, tf.float32)
        loss_cls_pos = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(gt_cls, pred_cls) * float_positive_mask
        ) / batch_size
        num_positive = tf.reduce_sum(tf.cast(bool_positive_mask, tf.int32))

        # hard negative mining
        bool_negative_mask_without_mining = float_positive_mask < 0.5
        float_negative_mask_without_mining = tf.cast(bool_negative_mask_without_mining, tf.float32)
        n_class = tf.shape(pred_cls)[-1]
        gt_cls_neg = tf.ones_like(gt_cls) * (n_class - 1)
        loss_cls_neg_without_mining = tf.nn.sparse_softmax_cross_entropy_with_logits(gt_cls_neg, pred_cls)
        loss_cls_neg_without_mining *= float_negative_mask_without_mining  # apply negative mask to remove positive ones
        mining_top_k, _ = tf.math.top_k(tf.reshape(loss_cls_neg_without_mining, (-1)), num_positive * 3)
        mining_top_k_min = mining_top_k[-1]
        assert mining_top_k_min == tf.reduce_min(mining_top_k)
        # generate final negative mask
        bool_negative_mask = loss_cls_neg_without_mining >= mining_top_k_min
        float_negative_mask = tf.cast(bool_negative_mask, tf.float32)
        # assert tf.reduce_sum(float_negative_mask) == tf.reduce_sum(float_positive_mask) * 3 # not always work
        assert tf.reduce_sum(tf.cast(tf.logical_and(bool_positive_mask, bool_negative_mask), tf.float32)) == 0.0

        # cls loss negative
        loss_cls_negative = tf.reduce_sum(
            loss_cls_neg_without_mining * float_negative_mask
        ) / batch_size

        # loc loss
        float_gt_mask = tf.cast(gt_mask, tf.float32)
        loss_box = tf.reduce_sum(
            tf.reduce_sum(tf.abs(pred_box - gt_box), axis=-1) * float_gt_mask
        ) / batch_size

        # logger.debug("masked box gt: %s", tf.boolean_mask(gt_box, gt_mask)[0])
        # logger.debug("masked box pred: %s", tf.boolean_mask(pred_box, gt_mask)[0])
        # logger.debug("Label | Pred: %s",
        #              tf.concat([tf.boolean_mask(gt_box, gt_mask), tf.boolean_mask(pred_box, gt_mask)], axis=-1))

        logger.debug("Loss function loc loss: %s | cls loss: %s | cls loss negative: %s | total loss: %s",
                     loss_box.numpy(), loss_cls_pos.numpy(), loss_cls_negative.numpy(),
                     (loss_box + loss_cls_pos + loss_cls_negative).numpy())

        loss_info = {"cls loss pos": (batch_size * loss_cls_pos / tf.reduce_sum(float_positive_mask)).numpy(),
                     "cls loss neg": (batch_size * loss_cls_negative / tf.reduce_sum(float_negative_mask)).numpy(),
                     "loc loss": (batch_size * loss_box / tf.reduce_sum(float_gt_mask)).numpy()}

        return loss_box + loss_cls_pos + loss_cls_negative, loss_info

    def show_summary(self):
        self._model.summary()
        tf.keras.utils.plot_model(self._model,
                                  to_file=str(self.__class__.__name__) + ".png",
                                  show_shapes=True,
                                  dpi=50)

    def save(self, path="model_weight.h5"):
        self._model.save(path)
        logger.info("Model is saved to %s", path)

    def load(self, path="model_weight.h5"):
        self._model = tf.keras.models.load_model(path)
        logger.info("Model is loaded from %s", path)

    def get_tf_model(self):
        return self._model

    def get_prior_box(self):
        return self._prior_box

    def visualize_prior_box(self, name="ssd visualize"):
        cx_pre = self._prior_box[0][0]
        cy_pre = self._prior_box[0][1]
        image = np.zeros([300, 300, 3], dtype=np.uint8)
        for cx, cy, w, h in self._prior_box:
            if cx != cx_pre or cy != cy_pre:
                cv2.imshow(name, image)
                ch = cv2.waitKey(0)
                if ch == "q":
                    break
                image = np.zeros([300, 300, 3], dtype=np.uint8)
                cx_pre = cx
                cy_pre = cy

            cv2.rectangle(
                image,
                (int((cx - w / 2) * self._INPUT_SHAPE[1]), int((cy - h / 2) * self._INPUT_SHAPE[0])),
                (int((cx + w / 2) * self._INPUT_SHAPE[1]), int((cy + h / 2) * self._INPUT_SHAPE[0])),
                (255, 255, 255)
            )

    def visualize_dataset(self, image, gt_cls, gt_bbox, mask,
                          score=None, name="ssd visualize", show=False, label_names=None, label_colors=None):
        image = np.array(((image / 2 + 0.5) * 255).numpy(), dtype=np.uint8)
        gt_bbox, gt_cls = np.array(gt_bbox.numpy()), np.array(gt_cls.numpy())
        mask = np.array(mask)

        if len(image.shape) > 3:
            image = image[0]
            gt_cls = gt_cls[0]
            gt_bbox = gt_bbox[0]
            mask = mask[0]
            if score is not None:
                score = score[0]

        gt_bbox_masked = gt_bbox[mask]
        gt_cls_masked = gt_cls[mask]
        default_box_masked = self._prior_box[mask]
        if score is None:
            score = np.ones_like(gt_cls_masked)
        else:
            score = np.array(score)[mask]
        for s, cls, bbox, default_box in zip(score, gt_cls_masked, gt_bbox_masked, default_box_masked):
            # print(cls, bbox)
            cx, cy = (bbox[:2] * default_box[2:] + default_box[:2]) * 300
            w, h = np.exp(bbox[2:]) * default_box[2:] * 300
            # print(image, cx, cy, w, h)
            cv2.rectangle(image, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)),
                          label_colors[int(cls)])
            if label_names is not None and label_colors is not None:
                # print(cls)
                cv2.putText(image, str(label_names[int(cls)]) + " " + str(s), (int(cx - w / 2), int(cy - h / 2) - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            label_colors[int(cls)], 1)

        if show:
            cv2.imshow(name, image)
            cv2.waitKey(1)

        return image

    def visualize(self, image, pred_conf, pred_bbox,
                  thresh=0.5, name="ssd visualize", show=False, mask=None, label_names=None, label_colors=None):
        pred_bbox = tf.tanh(pred_bbox)
        pred_conf = tf.nn.softmax(pred_conf)
        pred_score = None
        if mask is None:
            pred_score = tf.reduce_max(pred_conf[..., :-1], axis=-1)
            mask = pred_score > thresh
            mask_bg = pred_conf[..., -1] > thresh
            mask = tf.logical_and(mask, tf.logical_not(mask_bg))
        else:
            pred_conf = pred_conf[..., :-1]
        pred_cls = tf.argmax(pred_conf, axis=-1)
        return self.visualize_dataset(image, pred_cls, pred_bbox, mask, score=pred_score,
                                      name=name, show=show, label_names=label_names, label_colors=label_colors)
