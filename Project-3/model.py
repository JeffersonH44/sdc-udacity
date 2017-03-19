from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.utils.visualize_util import plot
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import cv2
import pickle
import numpy as np

# parameters
epochs = 40
learning_rate = 0.0001
batch_size = 50
activation = 'tanh'
brightness = 30


def process_image(img):
    # image crop from height (60 pixels from above and 25 below)
    img = img[60:-25, :, :]
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # image resize using Lanczos interpolation
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    # add brightness to the image but clamping those numbers to 255
    img = np.where((255 - img) < brightness, 255, img + brightness)

    return img


def generator(files, outputs, batch_size=256, augment_data=True):
    num_samples = len(files)
    images = []
    angles = []
    current_batch = 0
    while True:  # Used as a reference pointer so code always loops back around
        files, outputs = shuffle(files, outputs)
        for i in range(0, num_samples):
            name = files[i]
            output = outputs[i]

            # read and process image
            center_image = cv2.imread(name, 1)
            center_image = process_image(center_image)

            # process output
            steer = np.float(output)

            images.append(center_image)
            angles.append(steer)
            current_batch += 1

            # augment dataset fliping the image and their corresponding output
            if augment_data and current_batch != batch_size:
                flipped_image = cv2.flip(center_image, 1)
                new_steer = np.float(-output)

                images.append(flipped_image)
                angles.append(new_steer)
                current_batch += 1

            if current_batch == batch_size:
                X_train = np.array(images)
                y_train = np.array(angles)

                X_train = np.expand_dims(X_train, axis=3)

                yield X_train, y_train
                ## restart these variables to get a new batch
                images = []
                angles = []
                current_batch = 0

# Neural network model
model = Sequential()
# model normalization was an attempt but I didn't get better results
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(128, 128, 1)))
model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), input_shape=(128, 128, 1)))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation(activation))
model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2)))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation(activation))
model.add(Flatten())
model.add(Dense(256, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(128, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(1, activation=activation))

plot(model, show_shapes=True, show_layer_names=False)
print(model.summary())

dataset = pickle.load(open("dataset.p", 'rb'))
X_train = dataset["data"]
y_train = dataset["output"]

# split dataset on train validation (80% training and 20% test)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape[0])
print("Validation set size:", y_train.shape[0])

adam = Adam(lr=learning_rate)
model.compile(optimizer=adam,
              loss='mse')

# in samples_per_epoch I multiply by 2 because I augment that dataset
history = model.fit_generator(generator=generator(X_train, y_train, batch_size=batch_size),
                              validation_data=generator(X_validation, y_validation, augment_data=False),
                              nb_epoch=epochs, samples_per_epoch=X_train.shape[0] * 2, nb_val_samples=2500)

model.save('model.h5')
