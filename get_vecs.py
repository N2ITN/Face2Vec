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


def save_pickle(limit=10000, load=True, test=False):

    c = []
    v = []

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
            # c.append(temp.euclidian1D + temp.euclidian)
            c.append(temp.tensor)
            v.append(category)

        except Exception as e:
            print(e)
            continue
        if len(c) > limit:
            break
    if load:

        c_old, v_old = load_pickle()
        c = c_old + c
        v = v_old + v

    arr = [c, v]

    print(type(arr))

    if not test:
        pickle.dump(arr, open("saveCZ.p", "wb"))
    else:
        return arr


def load_pickle():
    data = pickle.load(open("saveCZ.p", "rb"))
    return data


# print(np.array(load_pickle()[0]).shape)
# print(len(load_pickle()[0]))


def process_name(name):
    if "/" in name:
        name = name.split('/')[1]

    return ''.join([i for i in name.lower().replace('.jpg', '') if i.isalpha()])


def make_dictionary():
    files = glob("faces/*")
    uniqueSet = set()
    names = [uniqueSet.add(process_name(x)) for i, x in enumerate(files)]

    uniqueList = list(uniqueSet)
    uniqueList.sort()

    return {k: v for v, k in enumerate(uniqueList)}, files


def main():
    return make_dictionary()


names, files = main()


def rerun_files():
    s = save_pickle(load=False, test=False)


def examine_cached():
    X, y = load_pickle()

    print(X[0].shape)


# save_pickle(load=False, test=False )
# examine_cached()
