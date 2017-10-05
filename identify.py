""" Detect face with OpenCV, return DLib rectangle """

import cv2
import dlib
import numpy as np
from sklearn.preprocessing import MaxAbsScaler


class img_keypoints():

    def __init__(self, pic, draw=False):
        check_detectors()
        self.draw = draw
        self.img = cv2.imread(pic)
        self.img_name = pic

        if not hasattr(self.img, 'nbytes'):

            raise Exception(pic + ' is not a valid image path!')
        self.detect_face()
        self.detect_keypoints()
        self.show_keypoints()

    def geom(self):
        self.centerPoint = self.keypoints[30]
        self.all_euclidian()
        self.show_keypoints()

    def detect_face(self):
        """ Greyscale img, boost contrast, detect faces, extract coords from first face found, convert to dlib rectangle"""

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
        self.clahe_image = clahe.apply(gray)

        self.faceprime = face_cascade.detectMultiScale(self.clahe_image, 1.3, 5)
        self.write_img('equalized_img', cv2.equalizeHist(self.clahe_image))
        ''' 
        If opencv's Haar detector can't find a shape, it's likely dlib can if we pass along the 
        image. Otherwise, we use the cropped result from opencv 
        '''
        if len((self.faceprime)) == 0:
            self.faceprime = [[0, 0, self.img.shape[0], self.img.shape[1]]]
            self.color_normal = self.img
            self.equalized_img = cv2.equalizeHist(self.clahe_image)

        else:
            self.crop_image()

    def crop_image(self):
        """ Convert face coords to rectangle corner points, grow rectangle to capture full face """
        for (x, y, w, h) in self.faceprime:
            grow = h / 7
            size = (h, w)
            x1 = int(x - grow)
            x2 = int(x + w + grow)
            y1 = int(y - grow)
            y2 = int(y + h + grow)

        # Save cropped face img
        self.color_normal = self.img[y1:y2, x1:x2]

        clahe_crop = self.clahe_image[y1:y2, x1:x2]
        self.equalized_img = cv2.equalizeHist(clahe_crop)
        self.clahe = clahe_crop

    def detect_keypoints(self):
        # Detect face landmarks with dlib rectangle, dlib shape predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        dets = detector(self.equalized_img, 1)
        # shape = predictor(self.clahe,1)
        for k, d in enumerate(dets):

            # Get the landmarks/parts for the face in box d.
            shape = predictor(self.equalized_img, d)
            # Get the landmarks/parts for the face in box d.
            self.shape = predictor(self.img, d)

            self.keypoints = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]

        return self.keypoints

    def keypoints_mask(self):
        print(self.keypoints)

    def show_keypoints(self):
        # shape = self.shape
        for i in range(1, 68):

            cv2.circle(self.color_normal, self.keypoints[i], 1, (0, 235, 235), thickness=2)
        self.write_img('draw', self.color_normal)

    def write_img(self, path, image):
        if not self.draw:
            return
        im_out = './' + path + '/' + self.img_name.split('/')[-1]
        cv2.imwrite(im_out, image)

    def euc_center(self):

        euclidian = [self.distance_2D(self.centerPoint, point) for point in self.keypoints]
        euclidian = np.array(euclidian).reshape(-1, 1)
        norm = MaxAbsScaler().fit_transform(euclidian)
        self.euclidian = norm

    def euc_xy(self):
        euclidian1D = []

        [euclidian1D.append(self.distance_1D(self.centerPoint, point)) for point in self.keypoints]

        x, y = [x for x in zip(*euclidian1D)]

        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        x = MaxAbsScaler().fit_transform(x)
        y = MaxAbsScaler().fit_transform(y)

        self.euclidianX = x
        self.euclidianY = y

    def all_euclidian(self):
        self.euc_xy()
        self.euc_center()
        self.tensor = np.rot90(np.hstack((self.euclidianX, self.euclidianY, dself.euclidian)))

    def distance_1D(self, a, b):
        """ Return x,y distance between 2 coordinate pairs, assistant function to array_maker """
        x1, y1 = a
        x2, y2 = b
        x = x1 - x2
        y = y1 - y2
        return x, y

    def distance_2D(self, a, b):
        """ Return euclidian distance between 2 coordinate pairs, assistant function to array_maker """
        x1, y1 = a
        x2, y2 = b
        a = np.array((x1, y1))
        b = np.array((x2, y2))
        dist = np.linalg.norm(a - b)

        return dist


def test():
    ex = img_keypoints('snapcrop.jpg')

    return ex


def check_detectors():
    """ Runs as import to check for detector files """
    import os
    from urllib.request import urlretrieve

    dl = {
        'haarcascade_frontalface_alt2.xml':
            'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml',
        'shape_predictor_68_face_landmarks.dat':
            'https://github.com/AKSHAYUBHAT/TensorFace/raw/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
    }

    for k in dl:
        if not os.path.exists(k):
            print('downloading', k)
            urlretrieve(dl[k], k)
