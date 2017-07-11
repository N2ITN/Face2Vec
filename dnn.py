import time
from random import shuffle
import numpy as np
from get_vecs import load_pickle, names_files, one_pic
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import define_model


def kFold(k):
    """ Orchestrate k-fold xval """
    n = y.shape[0]
    split = n // k
    resid = n % k

    chunks = [j for j in range(k)]
    shuffle(chunks)
    for i in chunks:
        start = i * split
        end = (i + 1) * split
        yield start, end


def subdivide(start, end):
    """ Split up sample groups for k-fold """
    X_test = X[start:end]
    first = X[:start]
    second = X[end:]
    X_train = np.vstack((first, second))

    y_test = y[start:end]
    first = y[:start]
    second = y[end:]
    y_train = np.vstack((first, second))
    return X_train, y_train, X_test, y_test


def run_kfold(load=False):
    """ Entry point for training """
    for start, end in kFold(10):
        X_train, y_train, X_test, y_test = subdivide(start, end)
        train_model(X_train, y_train, X_test, y_test, load=False)


def train_model(X_train, y_train, X_test, y_test, load=False):
    """ Train, save weights with automatic checkpointing based on time. Includes callbacks to tensorboard """

    filepath = "weights.best.hdf5"
    if load == True:
        try:
            model.load_weights(filepath)
        except OSError:
            print("model not reloaded")
        else:
            print("model loaded")

    model.compile(
        loss='categorical_crossentropy',
        optimizer='nadam',
        metrics=['accuracy']
    )

    now = time.strftime("%c")

    checkpoint = ModelCheckpoint(
        filepath, monitor='loss', verbose=0, save_best_only=False, mode='auto'
    )

    # tensorboard = TensorBoard(
    #     log_dir='./logs/' + now, histogram_freq=1, write_graph=True
    # )
    # callbacks_list = [checkpoint, tensorboard]
    callbacks_list = [checkpoint]

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=15,
        shuffle='batch',
        callbacks=callbacks_list,
        verbose=1
    )

    (loss, accuracy) = scores = model.evaluate(X_test, y_test, batch_size=15)

    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))


def predict(subject):
    """ Predict the class of a particular image """
    if isinstance(subject, str):
        try:
            X, y = one_pic(subject)
            label = y

        except RuntimeError:
            return
    else:
        X, y = subject

        X = np.expand_dims(X, axis=0)
        X = np.array(X).astype('float64')
        label = y

    model = define_model.model(n_classes())
    try:
        model.load_weights("weights.best.hdf5")
    except OSError:
        print("creating new weights file")

    pred = model.predict_classes(X, batch_size=1)

    pred = pred.tolist()[0]

    reverseDict = {i[1]: i[0] for i in names_files('train')[0].items()}

    print(pred, label)
    print(
        pred == label, 'prediction:', reverseDict[pred], 'actual:',
        reverseDict[label]
    )
    return (pred == label)


def get_accuracy():
    """ Run all results through prediction and get total accuracy """
    X, y = load_pickle('test')
    results = [predict((X[i], y[i])) for i in range(len(y))]
    if results.count(True) > 0:
        print((results.count(True) / len(y)) * 100, ' percent')
    else:
        print('zero percent')


def n_classes():
    X, y = load_pickle('train')

    return len(y)


if __name__ == '__main__':

    train = True
    if train:
        model = define_model.model(n_classes())
        X, y = load_pickle('train')
        X = np.array(X).astype('float64')
        y = np_utils.to_categorical(y, 0)
        run_kfold()

    def evaluate():
        get_accuracy()

    evaluate()
