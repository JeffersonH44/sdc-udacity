from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Merge
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import cv2
import pickle
import numpy as np

# parameters
epochs = 17
learning_rate = 0.0001
batch_size = 50
activation = 'tanh'


def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    # destination = np.zeros(img.shape)
    # img = cv2.normalize(img, destination, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = np.expand_dims(img, axis=0)
    return img


def generator(files, outputs, batch_size=256):
    num_samples = len(files)
    while True:  # Used as a reference pointer so code always loops back around
        files, outputs = shuffle(files, outputs)
        for offset in range(0, num_samples, batch_size):
            X_batch = files[offset:offset + batch_size]
            y_batch = outputs[offset:offset + batch_size]

            images = []
            angles = []
            for image, output in zip(X_batch, y_batch):
                name = image
                center_image = cv2.imread(name, 1)
                center_image = process_image(center_image)

                steer = np.float(output)

                images.extend(center_image)
                angles.append(steer)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train


model = Sequential()
model.add(Convolution2D(24, 3, 3, border_mode='valid', subsample=(2, 2), input_shape=(128, 128, 3)))
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
model.add(Dense(1, activation="tanh"))
dataset = pickle.load(open("dataset.p", 'rb'))
X_train = dataset["data"]
y_train = dataset["output"]

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

adam = Adam(lr=learning_rate)
model.compile(optimizer=adam,
                       loss='mse')
history = model.fit_generator(generator=generator(X_train, y_train, batch_size=batch_size),
                                       validation_data=generator(X_validation, y_validation),
                                       nb_epoch=epochs, samples_per_epoch=X_train.shape[0], nb_val_samples=X_validation.shape[0])

model.save('my_model.h5')
