from keras.layers import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization


def model(n_classes):
    """ Definition of the model """
    model = Sequential()
    model.add(Dense(68, input_shape=(3, 68,), kernel_initializer='TruncatedNormal'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Dense(32))
    model.add(Activation('sigmoid'))

    model.add(Flatten())
    model.add(Dense(n_classes))

    model.add(Activation('softmax'))
    return model
