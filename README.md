Face2Vec

```
sudo apt liboost-all-dev
sudo pip install opencv-python
sudo pip install dlib
```

Additional OpenCV, DLib classifiers:
https://drive.google.com/open?id=0B8yndZciEDldWEFQc1lRZ3J6ZlE


use coarser yet faster methods than Inceptionv3 on a GPU

### Pre-Processing Pipeline
  Using the classical computer vision libraries OpenCV2 and Dlib, image are searched for faces. Any faces present are cropped out, and labled wit ha key point detector. These keypoints are stored in 2-dimensional array of size (2,68) representing the euclidian coordinates of each keypoint.

#### Clarity boosting:
   Images are greyscaled, and contrast boosted with histogram equalization. This technique will make images clearer by scaling the changes in brightness more evently accross the image.
  
### Face detection & Cropping:
  Next the image is scanned for faces using a Haar Cascade classifier within the Viola-Jones detection algorithm. Developed in 2001, this was the first real-time face detection technique, and has the advantage of being much faster than convolutional neural networks of toady to the detriment of accuracy, versatility, and it's general ability to handle  . As articulated in a more recent paper (https://arxiv.org/pdf/1408.1656.pdf) on mathematical computer vision techniques: "in unconstrained scenes such as faces in a crowd", The Viola Jones fails to "perform well due to large pose variations, illumination variations, occlusions, expression variations, out-of-focus blur, and low image resolution." As the goal of this tutorial is to make a lightweight face recognition algorithm, a classifier that is able to recognize the presence of a face quickly in a moderate diversity of contexts will suffice.
  
 ### Facial landmarks
  The dlib face detector scans the cropped face images to recognize  for images using a more specialized classfier, which as described by the author "is made using the classic Histogram of Oriented Gradients (HOG) feature combined with a linear classifier, an image pyramid, and sliding window detection scheme." http://dlib.net/face_landmark_detection.py.html The HOG based detectors may not be as robust for finding high contrast features as Haar detectors, but they have the advantage of recognizing variations in shading. For this reason, they are implemented as part of dlib's facial keypoint identification.
But why use two face recognizers?? The Viola Jones recognizer is faster and more accurate at finding a face in a large image space. To find the landmarks, the HOG is mantatory, but using the Viola Jones method first and cropping to the face it finds allows for decreased processing time, because the slower HOG is only looking at a small area.

What is created by facial marks is a standarized 'face mask' of 68 points corresponding to points on each face, like eyebrows, top of the nose, etc. The keypoints are stored in an array associated with a face label.

[1]/_DRAW_faces/Carrie1.jpg
