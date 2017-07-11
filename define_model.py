from keras.layers import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.regularizers import l2


def model(n_classes):
    """ Definition of the model """
    model = Sequential()
    model.add(Dense(408, input_shape=(2, 68,), kernel_initializer='uniform'))
    model.add(Activation('relu'))

    model.add(
        Dense(
            204,
            kernel_initializer='uniform',
            activity_regularizer=l2(0.003),
            bias_regularizer=l2(0.003)
        )
    )
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(
        Dense(
            51,
            kernel_initializer='uniform',
            activity_regularizer=l2(0.003),
            bias_regularizer=l2(0.003)
        )
    )
    model.add(Activation('relu'))

    model.add(
        Dense(
            n_classes,
            kernel_initializer='uniform',
            activity_regularizer=l2(0.003),
            bias_regularizer=l2(0.003)
        )
    )
    model.add(Activation('softmax'))
    return model
