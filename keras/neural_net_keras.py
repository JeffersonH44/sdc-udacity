from urllib.request import urlretrieve
from os.path import isfile
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

import cv2
import pickle
import tensorflow as tf


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


if not isfile('train.p'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Train Dataset') as pbar:
        urlretrieve(
            'https://s3.amazonaws.com/udacity-sdc/datasets/german_traffic_sign_benchmark/train.p',
            'train.p',
            pbar.hook)

if not isfile('test.p'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Test Dataset') as pbar:
        urlretrieve(
            'https://s3.amazonaws.com/udacity-sdc/datasets/german_traffic_sign_benchmark/test.p',
            'test.p',
            pbar.hook)

print('Training and Test data downloaded.')

tf.python.control_flow_ops = tf

print('Modules loaded.')

with open('train.p', 'rb') as f:
    data = pickle.load(f)

    # TODO: Load the feature data to the variable X_train
    X_train = data['features']
    # TODO: Load the label data to the variable y_train
    y_train = data['labels']

X_train, y_train = shuffle(X_train, y_train)


# TODO: Normalize the data features to the variable X_normalized

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    # print(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + (((image_data - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))


X_normalized = normalize_grayscale(X_train)

lb = LabelBinarizer()
lb.fit(y_train)

y_one_hot = lb.transform(y_train)

# TODO: Build a model
model = Sequential()

model.add(Convolution2D(32, 2, 2, border_mode='valid', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation('tanh'))
model.add(Convolution2D(64, 2, 2, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation('tanh'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('tanh'))
model.add(Dense(43))
model.add(Activation('softmax'))
# TODO: Compile and train the model

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, batch_size=64, nb_epoch=30, validation_split=0.2)

# TODO: Load test data
with open('test.p', 'rb') as f:
    data = pickle.load(f)
    X_test = data['features']
    y_test = data['labels']

X_normalized_test = normalize_grayscale(X_test)
y_one_hot_test = lb.transform(y_test)

measures = model.evaluate(X_normalized_test, y_one_hot_test)
for i in range(len(measures)):
    metric_name = model.metrics_names[i]
    metric_value = measures[i]
    print('{}: {}'.format(metric_name, metric_value))