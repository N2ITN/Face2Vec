import cv2
import sys
import os
import logging as log
import datetime as dt
from time import sleep
import argparse
from urllib.request import urlretrieve
import copy
from identify import check_detectors

#TODO: Implement Args
#TODO: Aligninng
#TODO: Implement Augmentation


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--Name", required=True,
    help="Name of the person for which the data is collected")
ap.add_argument("-d", "--dataset", type=str, default="train",
    help="Name of the folder containing the dataset")
ap.add_argument("-m", "--method", type=int, default=0,
    help="Method of augmentation, 0 for webcam 1 for synthetic")
ap.add_argument("-f", "--files", type=int, default=20,
    help="Number of files desired")
args = vars(ap.parse_args())
check_detectors()

if args['method']==0:
    web_aug(args['Name'])
else:
    from dataag import augment
    c,l=check_name(args['Name'],args['dataset'])

    augment(c,l,args['files'])
def web_aug():
    cascPath = "haarcascade_frontalface_alt2.xml"

    video_capture = cv2.VideoCapture(0)

    c,_=check_name(arg['Name'],args['dataset'])
    # anterior = 0


    faceCascade = cv2.CascadeClassifier(cascPath)

    c=check_name(args_name)
    orig_c=c
    while c-orig_c<=args['files']:
        sleep(30)
        if not video_capture.isOpened():
            print('Unable to load camera.')

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        clean_frame=copy.deepcopy(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )


        # print(faces)

        if faces!=():
            cv2.imwrite(clean_frame,args['dataset']+'{}_{}'.format(name,c))
            c+=1



        # Draw a rectangle around the faces

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


        # if anterior != len(faces):
        #     anterior = len(faces)
        #     log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

        # Display the resulting frame
        cv2.imshow('Video', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def check_name(name,path):
    nlist=[]
    for i in os.listdir(path):
        if name in i:
            nlist.append(i.split('.')[0])
    return max([int(i[len(name):]) for i in nlist]),[path+'/'+i for i in nlist]







