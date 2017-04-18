## Writeup 

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Running the whole model

* Extract the dataset on the dataset folder.
* Run the following commands:

```bash
python get_data.py
python training_step.py
python video_prediction.py # or python video_prediction_1.py
```
* See video on the output images folder.


[//]: # (Image References)
[image0]: output_images/car_not_car.png
[image1]: output_images/original.png
[image2]: output_images/augmentation.png
[image3]: output_images/channels.png
[image4]: output_images/spatial.png
[image5]: output_images/channel1.png
[image6]: output_images/channel2.png
[image7]: output_images/channel3.png
[image8]: output_images/hog1.png
[image9]: output_images/hog2.png
[image10]: output_images/hog3.png
[image11]: output_images/features.png
[image12]: output_images/features_norm.png
[image13]: output_images/window_scheme.png
[image14]: output_images/threshold_heatmap.png
[image15]: output_images/prediction.png
[video1]: output_images/svm_output.mp4
[video2]: output_images/ssd_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points  

---

## SVM and HOG features

#### Data reading and augmentation

First I read all the images (64x64) that contains `vehicle` and `non-vehicle` samples, 
to avoid overfitting on the classifier because the images were extracted from 
time-line series, I took every 10 images from the dataset 
(lines 21 through 29, file `get_data.py`).  
 
Also we have to take in count that the images are on png format, so I should rescale 
those images to a range between 0 and 255.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image0]

Then because of the few images that I got, I decided to augment the dataset 
by flipping the images, as the example shows (line 229, file `lesson_functions.py`):

![alt text][image2]

#### Feature extraction

Then I extract all the features given the `params.py` file:

```python
# for color space
color_space = 'YCrCb'
# spatial size transform
spatial_feat = True
spatial_size = (32, 32)
# histogram
hist_feat = True
hist_bins = 32
# HOG
hog_feat = True
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'
```

* I convert to 'YCrCb' space (line 158, file `lesson_functions.py`)

![alt text][image3]

* I got spacial features from a resized image (from 64x64 to 32x32)
(line 167, file `lesson_functions.py`)

![alt text][image4]

* I got the color histogram from all channels of the image (line 172, 
file `lesson_functions.py`) 

![alt text][image5]
![alt text][image6]
![alt text][image7]

* I got the HOG features, I tried several combinations of parameters
with the classifier, using only the first channel but at the end
using all the channel was the best options to get better results with
the classifier (lines 189-206, file `lesson_functions.py`).

Also to improve the time detection I used the HOGDescriptor provided
by openCV instead of the skimage implementation as you can see in 
[this](https://discussions.udacity.com/t/good-tips-from-my-reviewer-for-this-vehicle-detection-project/232903/7) forum.

```ipnbpython
In [12]: %timeit -n100 -r1 opencv_hog.compute(image)
100 loops, best of 1: 204 us per loop
In [13]: %timeit -n100 -r1 hog(image)
100 loops, best of 1: 6.27 ms per loop
```

Images from the hog features:

![alt text][image8]
![alt text][image9]
![alt text][image10]

At the end I concatenate all these features (spatial, color and hog),
here's is an image of these features without normalization
(lines 210, file `lesson_functions.py`):

![alt text][image11]

and with normalization using StandardScaler (lines 36-37, 
file `get_data.py`):

![alt text][image12]

#### Training step:

I used two classifiers with the GridSearch approach to get the best parameters (`training_step.py`):

* **Gradient Boosting**: with the following parameters

XGBClassifier(n_estimators=25, learning_rate=0.5, nthread=8, objective='binary:logistic')

and I got the following results:

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| No car    | 1.0       | 1.0    | 1.0      | 383     |
| car       | 1.0       | 1.0    | 1.0      | 328     |
| avg/total | 1.0       | 1.0    | 1.0      | 711     |

* **LinearSVC**: with the following parameters

LinearSVC(loss='hinge', C=1.0)

and I got the following results:

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| No car    | 0.99      | 0.99   | 0.99     | 383     |
| car       | 0.99      | 0.99   | 0.99     | 328     |
| avg/total | 0.99      | 0.99   | 0.99     | 711     |

Even when the Gradient Boosting had better results on the test set, when I applied 
to the videos I got better results with the LinearSVC.

#### Sliding window and heat map

I used a pyramidal scheme given the appearance of the cars on the videos,  
'small' windows (64x64) on the top of the region that cars appears and 'bigger'
windows (128x128) below of the region, I made this to reduce the search space 
over the image, this windows has an overlap of 0.6.  the scheme is shown in the 
following image (lines 77-86, `video_prediction.py`)

![alt text][image13]

To avoid the false positives detection I used the heat map approach proposed in the lessons,
I update every 7 frames and I threshold the heat map with 6 frames (lines 88, 94, `video_prediction.py`):

![alt text][image14]

for combining overlapping bounding boxes I used the `scipy.ndimage.measurements.label()`
over the thresholded heat-map to get the final bounding boxes (line 51-69, `video_prediction.py`),
the final prediction image:

![alt text][image15]

A project video output is here:

![alt text][video1]

## Single shot multibox detection (SSD)

I tried a deep learning approach to see if it's more faster, but this model is faster only on a good GPU.
This model is easy to feed, just take an image and resize it to 300x300 and put it to the neural network to
get the predictions, then apply some post-processing to get the bounding boxes to the original image.
 
The main idea of this network instead of using a sliding window is to discretize the whole image space with
some prior boxes that you can move and resize to the ground truth to make the predictions, the main advantage
of this approach that you reduce the search space that you can get with the sliding window if you want to detect
different kind of objects that have different bounding boxes sizes.

I took [this](https://github.com/cory8249/ssd_keras) implementation and adapt it to the video prediction 
(source `video_prediction_1.py`) just to make more confident with the predictions I applied the heat map
approach because the pre-trained model that I used is not working well. 
([Pretrained model](https://mega.nz/#F!7RowVLCL!q3cEVRK9jyOSB9el3SssIA) download `weights_SSD300.hdf5`) 

Here is the output video:

![alt text][video2]

---

### Discussion

* One of the main problem that I got with the training was the overfitting, because I used the whole dataset 
  that comes from a time series of images, and when I tried to detect the white car on the project_video
  was really hard, to deal with this problem I just take every 10 images on the dataset, was a really simple
  solution to deal with overfitting.

* In the current implementation I extract HOG features from each window instead from the whole image and extract
  the window, this is a big bottleneck on the prediction step that can be improved, but I also improve
  the HOG extraction by changing from skimage to OpenCV implementation

* The sliding window is really a big limitation of the model because if you want to detect object of different sizes,
  you have to go again through all the image to detect it, that's why I made some exploration with deep learning 
  approaches like SSD to make the model more robust. 

