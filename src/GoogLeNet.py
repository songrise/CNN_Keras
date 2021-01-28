"""
GoogLeNet(2014)
Reference:
"""


from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, Lambda, GlobalAveragePooling2D, concatenate, Input
from keras.models import Sequential
from tensorflow.nn import local_response_normalization  # LRN layer
from keras.models import Model


def inception_module(input_shape=None, kernel_number=None):
    """
    Input: 
        input_shape: shape of input tensor
        kernel_number: a list object with 6 elements, each corresponds to the number of kernels in convolution layer
    Return: 
        construct and return an inception module (keras Model)
    """
    NUM_OF_CONV = 6
    if kernel_number is None or len(kernel_number) != NUM_OF_CONV:
        raise ValueError("Wrong input for parameter 'kernel_number'")
    input_ = Input(shape=input_shape)  # to be edited

    num_ker_1, num_ker_2, num_ker_3, num_ker_4, num_ker_5, num_ker_6 = kernel_number

    branch_a = Conv2D(num_ker_1, 1,
                      activation='relu', strides=1, padding="same")(input_)
    branch_b = Conv2D(num_ker_4, 1, activation='relu',
                      strides=1, padding="same")(input_)
    branch_b = Conv2D(num_ker_2, 3, activation='relu',
                      strides=1, padding="same")(branch_b)

    branch_c = Conv2D(num_ker_6, 1, activation='relu',
                      strides=1, padding="same")(input_)
    branch_c = Conv2D(num_ker_3, 5, activation='relu',
                      strides=1, padding="same")(branch_c)

    branch_d = MaxPool2D(3, strides=1, padding="same")(input_)

    branch_d = Conv2D(num_ker_4, 1, activation='relu',
                      strides=1, padding="same")(branch_d)

    output = concatenate(
        [branch_a, branch_b, branch_c, branch_d], axis=-1)

    module_ = Model(inputs=[input_], outputs=[output])
    return module_


def google_net():
    """
    Construct and return a GoogleNet
    """
    conv_1 = Conv2D(64, kernel_size=(7, 7), padding="same",
                    strides=2, input_shape=[224, 224, 3], activation='relu')
    pool_2 = MaxPool2D(pool_size=(3, 3), padding="same", strides=2)

    lrn_3 = Lambda(lambda x: local_response_normalization(x))
    conv_4 = Conv2D(64, kernel_size=(1, 1), strides=1,
                    padding="same", activation='relu')
    conv_5 = Conv2D(192, kernel_size=(3, 3), strides=1,
                    padding="same", activation='relu')
    lrn_6 = Lambda(lambda x: local_response_normalization(x))
    pool_7 = MaxPool2D(pool_size=(3, 3), padding="same", strides=2)
    inc_8 = inception_module((28, 28, 192), [64, 128, 32, 32, 96, 16])
    inc_9 = inception_module((28, 28, 256), [128, 192, 96, 64, 128, 32])
    pool_10 = MaxPool2D(pool_size=(3, 3), strides=2, padding="same")
    inc_11 = inception_module((14, 14, 480), [192, 208, 48, 64, 96, 16])
    inc_12 = inception_module((14, 14, 512), [160, 224, 64, 64, 112, 24])
    inc_13 = inception_module((14, 14, 512), [128, 256, 64, 64, 128, 24])
    inc_14 = inception_module((14, 14, 512), [128, 288, 64, 64, 144, 32])
    inc_15 = inception_module((14, 14, 544), [256, 320, 128, 128, 160, 32])
    pool_16 = MaxPool2D(pool_size=(3, 3), strides=2, padding="same")
    inc_17 = inception_module((7, 7, 832), [256, 320, 128, 128, 160, 32])
    inc_18 = inception_module((7, 7, 832), [384, 384, 128, 128, 192, 48])
    pool_19 = GlobalAveragePooling2D()
    drop_20 = Dropout(0.40)
    fc_21 = Dense(1000, activation="softmax")

    model = Sequential([conv_1, pool_2, lrn_3, conv_4,
                        conv_5, lrn_6, pool_7, inc_8, inc_9,
                        pool_10, inc_11, inc_12, inc_13, inc_14,
                        inc_15, pool_16, inc_17, inc_18, pool_19,
                        drop_20, fc_21], name="GoogLeNet")

    model.compile(optimizer="adam",
                  loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    return model


if __name__ == "__main__":
    google_net().summary()
