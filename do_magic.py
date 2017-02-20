""" Extracts eye closure ratio data from crop2face output image"""
import cv2
import dlib
import glob
import os
import numpy
from time import time
picture_dir = ''
from identify import crop2face

#from memory_profiler import profile


def distance_2D(x1, y1, x2, y2):
    """ Return euclidian distance between 2 coordinate pairs, assistant function to array_maker """
    a = numpy.array((x1, y1))
    b = numpy.array((x2, y2))
    dist = numpy.linalg.norm(a - b)
    return dist


#@profile
def analyze(shape, frame):
    """ Calculate eye closure from landmarks """

    def array_maker(n1, n2):
        """ Extract point pair from DLib shape to array, calculate euclidian distance  """
        return distance_2D(
            shape.part(n1).x,
            shape.part(n1).y, shape.part(n2).x, shape.part(n2).y)

    left_len = array_maker(36, 39)
    left_open_top = array_maker(37, 41)
    left_open_bottom = array_maker(38, 40)
    left_open = (left_open_top + left_open_bottom) / 2.
    ratio_L = (left_open / left_len)
    ratio_L.round(1)

    right_len = array_maker(42, 45)
    right_open_top = array_maker(43, 47)
    right_open_bottom = array_maker(44, 46)
    right_open = (right_open_top + right_open_bottom) / 2.
    ratio_R = (right_open / right_len)
    ratio_R.round(1)
    total_ratio = (ratio_R + ratio_L) / 2

    # Draw landmarks on image and save, for demonstration / debug use only '''
    # TODO: Functionalize this
    for i in range(1, 68):
        cv2.circle(
            frame, (shape.part(i).x, shape.part(i).y),
            1, (255, 255, 0),
            thickness=1)
    # y1 = shape.part(8).y
    # y2 = (shape.part(24).y + shape.part(19).y) / 2
    # x1 = shape.part(2).x
    # x2 = shape.part(16).x
    # crop = frame[y2 - 60:y1 + 30, x1 - 50:x2 + 50]
    # cv2.imwrite('snap_draw.jpg', crop)
    cv2.imwrite('snap_draw.jpg', frame)

    # Return final ratio
    return total_ratio.round(4)
