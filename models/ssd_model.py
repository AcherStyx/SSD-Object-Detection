import logging
import tensorflow as tf

from tensorflow.keras import layers, Model
from templates import *

logger = logging.getLogger(__name__)


class SSDObjectDetectionModel(ModelTemplate):
    def __init__(self,
                 classes,
                 input_shape,
                 base_model_output_shape=(38, 38, 512)):
        self.INPUT_SHAPE = input_shape
        self.BASE_MODEL_OUTPUT_SHAPE = base_model_output_shape
        self.CLASSES = classes

        # check parameters
        try:
            assert len(base_model_output_shape) == 3
        except AssertionError:
            logger.warning("Input shape should have 3 dimension")

        self.FILTER_SIZE_1 = 4 * (self.CLASSES + 4)
        self.FILTER_SIZE_2 = 6 * (self.CLASSES + 4)
        self.SAMPLE_SIZE = (self.CLASSES + 4)

        super(SSDObjectDetectionModel, self).__init__()

    def build_vgg_model(self):

        args_3_64 = {"filters": 64,
                     "kernel_size": (3, 3),
                     "padding": "SAME",
                     "activation": "relu"}
        args_3_128 = {"filters": 128,
                      "kernel_size": (3, 3),
                      "padding": "SAME",
                      "activation": "relu"}
        args_3_256 = {"filters": 256,
                      "kernel_size": (3, 3),
                      "padding": "SAME",
                      "activation": "relu"}
        args_3_512 = {"filters": 512,
                      "kernel_size": (3, 3),
                      "padding": "SAME",
                      "activation": "relu"}
        args_1_256 = {"filters": 128,
                      "kernel_size": (1, 1),
                      "padding": "SAME",
                      "activation": "relu"}
        args_1_512 = {"filters": 512,
                      "kernel_size": (1, 1),
                      "padding": "SAME",
                      "activation": "relu"}
        args_pool = {"pool_size": (2, 2), "strides": (2, 2), "padding": "SAME"}

        hidden_layer = input_layer = layers.Input(shape=self.INPUT_SHAPE)

        # hidden_layer = layers.Conv2D(**args_3_64)(hidden_layer)
        # hidden_layer = layers.Conv2D(**args_3_64)(hidden_layer)
        # hidden_layer = layers.MaxPool2D(**args_pool)(hidden_layer)
        #
        # hidden_layer = layers.Conv2D(**args_3_128)(hidden_layer)
        # hidden_layer = layers.Conv2D(**args_3_128)(hidden_layer)
        # hidden_layer = layers.MaxPool2D(**args_pool)(hidden_layer)
        #
        # hidden_layer = layers.Conv2D(**args_3_256)(hidden_layer)
        # hidden_layer = layers.Conv2D(**args_3_256)(hidden_layer)
        # hidden_layer = layers.Conv2D(**args_1_256)(hidden_layer)
        # hidden_layer = layers.MaxPool2D(**args_pool)(hidden_layer)
        #
        # hidden_layer = layers.Conv2D(**args_3_512)(hidden_layer)
        # hidden_layer = layers.Conv2D(**args_3_512)(hidden_layer)
        # hidden_layer = layers.Conv2D(**args_1_512)(hidden_layer)
        #
        # output_layer = hidden_layer
        #
        # org_vgg = Model(inputs=input_layer,
        #              outputs=output_layer,
        #              name="VGG")
        # # self.model = org_vgg
        # # self.show_summary(with_plot=True)
        # return org_vgg

        model = tf.keras.applications.VGG16(include_top=False, input_shape=(300, 300, 3))
        pre_trained_vgg = Model(inputs=model.input, outputs=model.get_layer("block3_conv3").output)(input_layer)

        hidden_layer = layers.MaxPool2D(**args_pool)(pre_trained_vgg)

        hidden_layer = layers.Conv2D(**args_3_512)(hidden_layer)
        hidden_layer = layers.Conv2D(**args_3_512)(hidden_layer)
        hidden_layer = layers.Conv2D(**args_1_512)(hidden_layer)

        return Model(inputs=input_layer, outputs=hidden_layer, name="Pre-trained_VGG")

    def build_ssd_model(self):

        base_network_output = layers.Input(shape=self.BASE_MODEL_OUTPUT_SHAPE)

        feature_map_1 = base_network_output

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
                                    activation="relu",
                                    padding="SAME")(feature_map_1)
        detection_1 = layers.Reshape((38 * 38 * 4, self.SAMPLE_SIZE))(detection_1)

        detection_2 = layers.Conv2D(filters=self.FILTER_SIZE_2,
                                    kernel_size=(3, 3),
                                    activation="relu",
                                    padding="SAME")(feature_map_2)
        detection_2 = layers.Reshape((19 * 19 * 6, self.SAMPLE_SIZE))(detection_2)

        detection_3 = layers.Conv2D(filters=self.FILTER_SIZE_2,
                                    kernel_size=(3, 3),
                                    activation="relu",
                                    padding="SAME")(feature_map_3)
        detection_3 = layers.Reshape((10 * 10 * 6, self.SAMPLE_SIZE))(detection_3)

        detection_4 = layers.Conv2D(filters=self.FILTER_SIZE_2,
                                    kernel_size=(3, 3),
                                    activation="relu",
                                    padding="SAME")(feature_map_4)
        detection_4 = layers.Reshape((5 * 5 * 6, self.SAMPLE_SIZE))(detection_4)

        detection_5 = layers.Conv2D(filters=self.FILTER_SIZE_1,
                                    kernel_size=(3, 3),
                                    activation="relu",
                                    padding="SAME")(feature_map_5)
        detection_5 = layers.Reshape((3 * 3 * 4, self.SAMPLE_SIZE))(detection_5)

        detection_6 = layers.Conv2D(filters=self.FILTER_SIZE_1,
                                    kernel_size=(3, 3),
                                    activation="relu",
                                    padding="SAME")(feature_map_6)
        detection_6 = layers.Reshape((1 * 1 * 4, self.SAMPLE_SIZE))(detection_6)

        output_layer = layers.Concatenate(axis=-2)([detection_1, detection_2, detection_3,
                                                    detection_4, detection_5, detection_6])

        return Model(inputs=base_network_output,
                     outputs=output_layer,
                     name="SSDObjectDetectionModel")

    def build(self, *args):
        input_layer = layers.Input(shape=self.INPUT_SHAPE, name="Input_Image")
        vgg = self.build_vgg_model()(input_layer)
        ssd = self.build_ssd_model()(vgg)

        self.model = Model(inputs=input_layer,
                           outputs=ssd)


if __name__ == '__main__':
    my_model = SSDObjectDetectionModel(input_shape=(300, 300, 3),
                                       base_model_output_shape=(38, 38, 512),
                                       classes=80)

    my_model.show_summary(with_plot=True)
