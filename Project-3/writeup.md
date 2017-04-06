# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./cnn-architecture-624x890.png "Nvidia model"
[image3]: ./cropped_image.png "Cropped image"
[image4]: ./grayscale.png "grayscale image"
[image5]: ./resize.png "resized image"
[image6]: ./brightness.png "brightness image"
[image7]: ./histogram.png "brightness image"
[image8]: ./flip.png "flipped image"

### Files Submitted

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results
* format.py loads the csv file and read all necessary data for the model.py
file (generates a pickle file "dataset.p")
* run1.mp4 containing a record of the first track
* run2.mp4 containing a record of the reverse first track

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car
can be driven autonomously around the track by executing
```sh
python drive.py my_model.h5
```

or if you have a laptop with optimus and you want to run the model
with GPU you can run it with:

 ```sh
optirun python drive.py my_model.h5
```

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

At the preprocess step I made the following steps:

* Take images from all cameras (center, left and right) and add a
correction angle for left and right cameras, in order to get more data
for the model (format.py, from line 8 to 19)

* Dataset augmentation by flipping the image from the center, here is an
image:

![alt text][image8]

With this and the previous step I got a histogram normally distributed
of the steering angle as expected:

![alt text][image7]

* Crop the images, 60 pixels from above and 25 from below to
get only an image of the road (model.py, line 23), here is an image:

![alt text][image3]

* Convert image to grayscale to reduce the complexity of the model (model.py, line 25),
here is an image:

![alt text][image4]

* Image resize to 128x128, this makes the model more comfortable with
the predictions because it make the road more smaller that the real one
(model.py, line 27), here is an image:

![alt text][image5]

* Brightness augmentation to deal with roads were the road is not clear,
for example, track 2 of the oldest simulator contains mountains that
produces a lot of shadow (model.py, line 29)(note: at the end it doesn't
work on that track), Here is an image:

![alt text][image6]

In the beginning I took the Nvidia's model as an starting point:

![alt text][image2]

Rather than using 66x200 images I used 128x128 as I did in the previous
project, using this approach I didn't get nice result so I started to
remove convolutional layers (I did this because Nvidia's model was
designed for 10 car controls and I only need two) until I saw that
the model takes his first curve, later, in the second curve doesn't work
well because it's a more close curve and the model predicts an small steering
angle, so the model overfits over the larger curve (this is the reason that
I recorded more laps from recovering than from the center of the road *Section 4*),
in order to deal with overfitting I reduce the fully connected layers to 3 layers
and adjust their size, also I put dropout layers on each layer of the network
(final model on *section 5*).

With this adjustments I get the model to runs well on track 1(run1.mp4)
and track 1 reverse (run2.mp4).

Because I attempt to make the model works on the second track, I need to consider
other variable for better car handle because this track contains very closed curves,
downhills, etc. So I added new inputs to the final model
(I explained how I added this input to the model in *Section 5*):

* Current Throttle.
* Current Steering Angle.
* Current Speed.

And I predict the new steering angle and the throttle, so I collect data in the same
way as the first track (*Section 4*) and added to the previous dataset but I didn't get good results,
maybe because in this track I should drive more slow but I will do this as a future work.

#### 2. Attempts to reduce overfitting in the model

* The data was divided on Train/Validation dataset (80% train, 20% validation),
I don't see necessary to split that data also on test because I can use
the simulator as my test dataset to see if it doesn't overfits.
* Also, to avoid overfitting I used Dropout layers (lines 97, 101, 108, 110)
on each layer of my model.

#### 3. Model parameter tuning

At the beginning I took the parameters of the previous project and the
way I proceed was a trial an error test (like a grid search) because each training process
took me about 15 min each, the following was the parameters used on the grid search:

* Optimization model: Adam (I only used this because it works, and
doesn't add new parameters)
* Activation functions: tanh, sigmoid, hard sigmoid, softmax, relu.
* Epochs : 15, 20, 30, 40, 50
* Learning rate: 0.0001, 0.001, 0.01, 0.1
* Batch size: 50, 100, 150, 200, 250
* Brightness for preprocess images: +10, +20, +30
* Correction handle for images taken from right and left camera: 0.1, 0.2

Getting better results on these parameters:

* Optimization model: Adam (model.py)
* Epochs : 50 (model.py)
* Learning rate: 0.0001 (model.py)
* Batch size: 250 (model.py)
* Brightness for preprocess images: +30 (model.py)
* Correction handle for images taken from right and left camera: 0.05 (preprocess.py)
* Activation: tanh (this one was the unique one that work, the other ones needs the camera
input normalized otherwise the model doesn't optimize(Gradient Vanishing), and even with this
doesn't get good results)

#### 4. Appropriate training data

For the training data I run the simulation and get the following data:

* 3 laps on track 1 driving on the center of the road (forward and backwards)
* 5 laps recovering from the right and left side of the road, just
recording the recovery part and not all the laps (forward and backwards)

For data augmentation I made a flip on X coordinates and multiply by -1
their corresponding steering angle.

It seems that I made a lot of laps but one of the problems with the latest simulator
is that takes much less photographs than the oldest simulator, so that make harder
the learning process but was nice to have this problem because in the beginning I
didn't know that is important the fps taken by the camera :).

#### 5. Final model (model.py lines 79-95)

I used the following model

For camera input

* Convolution 2D:
    * Stride: 2
    * Padding: valid
    * Filter size: 24
    + Convolution size: 5

* Pooling: avg
    * Size: 2

* Dropout
    * Probability: 0.5

* Activation Layer:
    kind = tanh

* Convolution 2D:
    * Stride: 2
    * Padding: valid
    * Filter size: 36
    + Convolution size: 5

* Pooling: avg
    * Size: 2

* Dropout
    * Probability: 0.5

* Activation Layer:
    kind = tanh

* Flatten layer:
    size: 1764

Here I combined the output of the convolutional layers with the inputs
(speed, throttle and steering angle).
* Merge Layer

* Fully connected:
    * Size: 256
    * Activation: Tanh

* Dropout
    * Probability: 0.5

* Fully connected:
    * Size: 128
    * Activation: Tanh

* Dropout
    * Probability: 0.5

* Fully connected (output layer):
    * Size: 1
    * Activation: Tanh

Here is a visualization of the final model:

![alt text][image1]