
"""
    AlexNet(2012)
    Reference: http://ethereon.github.io/netscope/#/gist/e65799e70358c6782b1b
"""

from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPool2D, Flatten
# for customize padding size.
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Sequential
from keras.optimizers import SGD

conv_1 = Conv2D(filters=96, kernel_size=(11, 11), strides=4,
                activation="relu", padding="valid", input_shape=[227, 227, 3])
norm_1 = BatchNormalization()  # LRN
pool_1 = MaxPool2D(pool_size=(3, 3), strides=2, padding="valid")
pad_1 = ZeroPadding2D(padding=(2, 2))
conv_2 = Conv2D(filters=256, kernel_size=(5, 5), strides=1,
                activation="relu")
norm_2 = BatchNormalization()
pool_2 = MaxPool2D(pool_size=(3, 3), strides=2, padding="valid")
pad_2 = ZeroPadding2D(padding=(1, 1))
conv_3 = Conv2D(filters=384, kernel_size=(3, 3), strides=1,
                activation="relu")
pad_3 = ZeroPadding2D(padding=(1, 1))
conv_4 = Conv2D(filters=384, kernel_size=(3, 3), strides=1,
                activation="relu")
pad_4 = ZeroPadding2D(padding=(1, 1))
conv_5 = Conv2D(filters=256, kernel_size=(3, 3), strides=1,
                activation="relu")
pool_3 = MaxPool2D(pool_size=(3, 3), strides=2, padding="valid")
flatten = Flatten()
fc_1 = Dense(units=4096, activation="relu")
fc_2 = Dense(units=4096, activation="relu")
drop = Dropout(0.5)
fc_3 = Dense(units=1000, activation="softmax")

AlexNet = Sequential([conv_1, norm_1, pool_1, pad_1, conv_2, norm_2, pool_2,
                      pad_2, conv_3, pad_3, conv_4,  pad_4, conv_5, pool_3, flatten, fc_1, fc_2, drop, fc_3])

AlexNet.compile(optimizer=SGD(learning_rate=0.001),
                loss="cross_entropy", metrics=["accuracy"])
AlexNet.summary()

pandas.DataFrame.to_csv()
