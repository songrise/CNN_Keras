"""
    FCN-8s model based on VGG-16 pretrained on ImageNet
    Reference: (TODO)
    Tensorflow 2.2.0
    J. Long, E. Shelhamer, T. Darrel,  Fully Convolutional Networks for Semantic Segmentation, arxiv preprint, 2015
"""

from keras.models import Model
from keras.layers import Conv2DTranspose, Add, Conv2D
from keras.applications import VGG16
from tensorflow.python.keras.layers.core import Activation


class FCN8(Model):
    def __init__(self, base_model: Model, classes: int = 1000, * args, **kwargs):
        """
            base_model: a keras.application.VGG16 object
            refer to call() for the connectivity information of each layer.
        """
        super().__init__(*args, **kwargs)

        self.vgg = base_model
        self.conv_6 = Conv2D(filters=1024, kernel_size=(
            7, 7), activation="relu", padding="same")
        self.conv_7 = Conv2D(classes, kernel_size=(1, 1), activation='relu',
                             padding='same', name="conv_7")
        self.conv_8 = Conv2D(classes, (1, 1), activation='relu',
                             padding='same', name="conv_8")
        self.conv_9 = Conv2D(classes, (1, 1), activation='relu',
                             padding='same', name="conv_9")

        self.deconv_1 = Conv2DTranspose(
            filters=classes, kernel_size=(2, 2), strides=2)
        self.deconv_2 = Conv2DTranspose(
            filters=classes, kernel_size=(2, 2), strides=2)
        self.deconv_3 = Conv2DTranspose(
            filters=classes, kernel_size=(8, 8), strides=8)

        self.merge_1 = Add()
        self.merge_2 = Add()
        self.act = Activation('softmax')

    def call(self, inputs, training=None, mask=None):
        """
            Annotations follows the paper (Long et al. 2015)
        """
        pool_3_out = self.vgg.get_layer('block3_pool').output
        pool_4_out = self.vgg.get_layer('block4_pool').output
        pool_5_out = self.vgg.get_layer('block5_pool').output

        conv_6_out = self.conv_6(pool_5_out)
        conv_7_out = self.conv_7(conv_6_out)

        # 2x upsampled of pool5 prediction
        deconv_1_out = self.deconv_1(conv_7_out)

        conv_8_out = self.conv_8(pool_4_out)  # pool4 prediction
        merge_1_out = self.merge_1([deconv_1_out, conv_8_out])

        # 2x upsampled prediction (before 16x upsample of FCN-16s)
        deconv_2_out = self.deconv_2(merge_1_out)

        conv_9_out = self.conv_9(pool_3_out)  # pool3 prediction
        merge_2_out = self.merge_2([conv_9_out, deconv_2_out])

        # 8x upsampled final prediction
        deconv_3_out = self.deconv_3(merge_2_out)
        output = self.act(deconv_3_out)
        return output


if __name__ == "__main__":
    vgg = VGG16(include_top=False, input_shape=(227, 227, 3))
    fcn = FCN8(base_model=vgg, classes=13)
    fcn.build(input_shape=(None, 227, 227, 3))
    fcn.summary()

    fcn.compile(loss='categorical_crossentropy',
                optimizer="adam", metrics=['accuracy'])
    fcn.summary()
    # import keras
    # keras.applications.VGG16(weights="imagenet").summary()
