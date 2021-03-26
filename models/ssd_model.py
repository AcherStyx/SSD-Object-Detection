import logging
import tensorflow as tf

from tensorflow.keras import layers, Model

logger = logging.getLogger(__name__)


class SSDObjectDetectionModel:
    def __init__(self,
                 classes,
                 input_shape=(320, 320, 3)):
        self.INPUT_SHAPE = input_shape
        self.CLASSES = classes

        # check parameters
        try:
            assert len(input_shape) == 3
        except AssertionError:
            logger.warning("Input shape should have 3 dimension")

        self.FILTER_SIZE_1 = 4 * (self.CLASSES + 4)
        self.FILTER_SIZE_2 = 6 * (self.CLASSES + 4)
        self.SAMPLE_SIZE = (self.CLASSES + 4)

        self._model = self._build()

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

        input_layer = layers.Input(shape=(None, None, 3))

        model = tf.keras.applications.VGG16(include_top=False)
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

        output_layer = [detection_1, detection_2, detection_3, detection_4, detection_5, detection_6]

        return Model(inputs=input_layer,
                     outputs=output_layer,
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
    def _loss(self, y_true, y_pred):
        loss = tf.constant(0.0, dtype=tf.float32)

        for image_true, image_pred in zip(y_true, y_pred):
            for featmap_true, featmap_pred in zip(image_true, image_pred):
                pass

    def train(self, dataset):
        pass

    def show_summary(self):
        self._model.summary()
        tf.keras.utils.plot_model(self._model,
                                  to_file=str(self.__class__.__name__) + ".png",
                                  show_shapes=True,
                                  dpi=50)

    def get_tf_model(self):
        return self._model


if __name__ == '__main__':
    my_model = SSDObjectDetectionModel(input_shape=(-1, -1, 3),
                                       classes=80)
    my_model.show_summary()
    dummy_input = tf.random.normal([1, 300, 300, 3])
    dummy_output = my_model.get_tf_model()(dummy_input)
    for i, out in enumerate(dummy_output):
        print("output {}: {}".format(i, out.shape))
