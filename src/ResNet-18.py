# """
# ResNet-18
# Reference:
# """


# from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, Lambda, GlobalAveragePooling2D, concatenate, Input
# from keras.models import Sequential
# from tensorflow.nn import local_response_normalization  # LRN layer
# from keras.models import Model


# def resnet_18():
#     """
#     Construct and return a GoogleNet
#     """
#     conv_1 = Conv2D(64, kernel_size=(7, 7), padding="same",
#                     strides=2, input_shape=[224, 224, 3], activation='relu')
#     pool_2 = MaxPool2D(pool_size=(3, 3), padding="same", strides=2)

#     pool_19 = GlobalAveragePooling2D()
#     drop_20 = Dropout(0.40)
#     fc_21 = Dense(1000, activation="softmax")

#     model = Sequential([conv_1, pool_2, lrn_3, conv_4,
#                         conv_5, lrn_6, pool_7, inc_8, inc_9,
#                         pool_10, inc_11, inc_12, inc_13, inc_14,
#                         inc_15, pool_16, inc_17, inc_18, pool_19,
#                         drop_20, fc_21], name="GoogLeNet")

#     model.compile(optimizer="adam",
#                   loss="cross_entropy", metrics=["cross_entropy_accuracy"])
#     return model


# if __name__ == "__main__":
#     resnet_18().summary()
