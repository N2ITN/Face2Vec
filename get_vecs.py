import hashlib
import pickle
from glob import glob
from random import shuffle

import numpy as np
from identify import img_keypoints


def one_pic(img):
    X = np.array([img_keypoints(img).tensor])
    name = process_name(img)
    y = names[name]
    return X, y


def save_pickle(fname, limit=10000, load=False):

    c = []
    v = []
    names, files = make_dictionary(fname)

    for i, x in enumerate(files):
        if i > limit:
            break
        print(i)
        name = process_name(x)
        category = names[name]
        print(x)
        try:
            temp = img_keypoints(x)
            temp.show_keypoints()
            c.append(temp.tensor)

            v.append(category)

        except Exception as e:
            print(e)
            continue
        if len(c) > limit:
            break
    if load:

        c_old, v_old = load_pickle(fname)
        c = c_old + c
        v = v_old + v

    arr = [c, v]

    pickle.dump(arr, open(fname + ".p", "wb"))


def load_pickle(name):
    return pickle.load(open(name + ".p", "rb"))


def process_name(name):
    if "/" in name:
        name = name.split('/')[-1]

    return ''.join([i for i in name.lower().replace('.jpg', '') if i.isalpha()])


def make_dictionary(name):
    files = glob("./" + name + "/*")
    uniqueSet = set()
    names = [uniqueSet.add(process_name(x)) for i, x in enumerate(files)]
    uniqueList = list(uniqueSet)
    uniqueList.sort()

    return {k: v for v, k in enumerate(uniqueList)}, files


def names_files(name):
    return make_dictionary(name)


def rerun_files():
    save_pickle(load=False, fname='test')
    save_pickle(load=False, fname='train')


def examine_cached():
    for i in ['train', 'test']:
        X, y = load_pickle(i)
        print(set([i.shape for i in X]))
        print(len(y))
        # print(X.shape)


if __name__ == '__main__':
    rerun_files()
    # print({i[1]: i[0] for i in names_files('train')[0].items()})
    # print({i[1]: i[0] for i in names_files('test')[0].items()})
    # examine_cached()
