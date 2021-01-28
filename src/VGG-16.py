"""
    VGG-16(2014)
    Reference:
"""

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
from keras.layers import Layer


def vgg_16():
    """
    build and return a VGG16 model
    """
    conv_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                    activation="relu", padding="same", input_shape=[224, 224, 3])

    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                    activation="relu", padding="same")

    pool_3 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")

    conv_4 = Conv2D(filters=128, kernel_size=(3, 3), strides=1,
                    activation="relu")
    conv_5 = Conv2D(filters=128, kernel_size=(3, 3), strides=1,
                    activation="relu")

    pool_6 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
    conv_7 = Conv2D(filters=256, kernel_size=(3, 3), strides=1,
                    activation="relu", padding="same")
    conv_8 = Conv2D(filters=256, kernel_size=(3, 3), strides=1,
                    activation="relu", padding="same")
    conv_9 = Conv2D(filters=256, kernel_size=(3, 3), strides=1,
                    activation="relu", padding="same")
    pool_10 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
    conv_11 = Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                     activation="relu", padding="same")
    conv_12 = Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                     activation="relu", padding="same")
    conv_13 = Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                     activation="relu", padding="same")
    pool_14 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
    conv_15 = Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                     activation="relu", padding="same")
    conv_16 = Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                     activation="relu", padding="same")
    conv_17 = Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                     activation="relu", padding="same")
    pool_18 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
    flat = Flatten()
    fc_19 = Dense(4096, activation="relu")
    fc_20 = Dense(4096, activation="relu")
    out_21 = Dense(1000, activation="softmax")

    VGG_16 = Sequential([conv_1, conv_2, pool_3, conv_4, conv_5, pool_6, conv_7,
                         conv_8, conv_9, pool_10, conv_11,  conv_12, conv_13, pool_14,
                         conv_15, conv_16, conv_17, pool_18, flat, fc_19, fc_20, out_21], name="VGG16")
    VGG_16.compile(optimizer="adam",
                   loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    return VGG_16


if __name__ == "__main__":
    vgg_16().summary()
