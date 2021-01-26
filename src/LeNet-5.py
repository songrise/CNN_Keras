"""
    AlexNet(1998) 
    Reference: http://yann.lecun.com/exdb/lenet/
    [LeCun et al., 1998]
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, november 1998.
"""

from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import AveragePooling2D
from keras.layers import Conv2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping

# The proprocessing of digit image is not included here.


def lenet_5():
    """
    build and return a LeNet-5 model
    """
    C1 = Conv2D(filters=6, kernel_size=(5, 5), padding="same",
                activation="tanh", input_shape=[28, 28, 1])  # since padding is applied, this would be same as inputing 32x32 image
    # In original implementation, subsampling should be take sum of 4 values in pooling window,
    # then multiply by a weight and add a trainable bias, after which the value passed to a sigmoid function
    S2 = AveragePooling2D(pool_size=(2, 2), strides=2)
    # According to Lecun(1998, p.8), S2 and C3 are not fully connect to 1)reduce connectivity 2)elimate symmetry.
    # This was simulated by a drop-out in my impl,
    drop = Dropout(0.5)
    C3 = Conv2D(filters=16, kernel_size=(5, 5),
                padding="valid", activation="tanh")
    S4 = AveragePooling2D(pool_size=(2, 2), strides=2)
    C5 = Conv2D(filters=120, kernel_size=(5, 5),
                padding="valid", activation="tanh")
    flat = Flatten()
    F6 = Dense(84, activation="tanh")

    out = Dense(10, activation="softmax")

    lenet_5 = Sequential(
        [C1, S2, drop, C3, S4, C5, flat, F6, out], name="LeNet-5")

    lenet_5.compile(loss='sparse_categorical_crossentropy',
                    optimizer="sgd", metrics=["accuracy"])

    return lenet_5


if __name__ == "__main__":
    lenet_5().summary()
