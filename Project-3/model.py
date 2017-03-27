from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Convolution2D, AveragePooling2D, Activation, Merge, Lambda
from keras.optimizers import Adam, Nadam
from keras.utils.visualize_util import plot
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import cv2
import pickle
import numpy as np

# parameters
epochs = 50
learning_rate = 0.0001
batch_size = 50
activation = 'tanh'
brightness = 10


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
    velocities = []
    angles = []
    current_batch = 0
    while True:  # Used as a reference pointer so code always loops back around
        files, outputs = shuffle(files, outputs)
        for i in range(0, num_samples):
            name = files[i][0]
            velocities.append([files[i][1], files[i][2], files[i][3]])

            steer = outputs[i][0]
            throttle = outputs[i][1]

            # read and process image
            center_image = cv2.imread(name, 1)
            if center_image is None:
                continue
            center_image = process_image(center_image)

            # process output
            steer = np.float(steer)
            throttle = np.float(throttle)

            images.append(center_image)
            angles.append([steer, throttle])
            current_batch += 1

            # augment dataset fliping the image and their corresponding output
            if augment_data and current_batch != batch_size:
                flipped_image = cv2.flip(center_image, 1)
                velocities.append([files[i][1], files[i][2], files[i][3]])
                new_steer = np.float(-steer)

                images.append(flipped_image)
                angles.append([new_steer, throttle])
                current_batch += 1

            if current_batch == batch_size:
                X_train = np.array(images)
                vels = np.array(velocities)
                y_train = np.array(angles)

                X_train = np.expand_dims(X_train, axis=3)

                # print(X_train.shape, vels.shape, y_train.shape)

                yield [X_train, vels], y_train
                ## restart these variables to get a new batch
                images = []
                angles = []
                velocities = []
                current_batch = 0

# speed input
speed_input = Input(shape=(3,), name='speed_input')

# camera input
camera_input = Input(shape=(128, 128, 1))
# lamb = Lambda(lambda x: (x / 255.0) - 0.5)(camera_input)
conv1 = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2))(camera_input)
pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
drop1 = Dropout(0.5)(pool1)
actv1 = Activation(activation)(drop1)
conv2 = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2))(actv1)
pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
drop2 = Dropout(0.5)(pool2)
actv2 = Activation(activation)(drop2)
conv_output = Flatten()(actv2)

# Merge inputs
merge = Merge(mode='concat')([conv_output, speed_input])
#dens1 = Dense(300, activation=activation)(merge)
#drop3 = Dropout(0.5)(dens1)
dens2 = Dense(200, activation=activation)(merge)
drop4 = Dropout(0.5)(dens2)
dens3 = Dense(100, activation=activation)(drop4)
drop5 = Dropout(0.5)(dens3)
main_output = Dense(2, activation=activation)(drop5)

model = Model(input=[camera_input, speed_input], output=[main_output])


"""
# Neural network model
model = Sequential()
# model normalization was an attempt but I didn't get better results
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, ))
model.add()
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation(activation))
model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2)))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation(activation))
model.add(Flatten())
model.add(Dense(300, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(150, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(1, activation=activation))
"""
dataset = pickle.load(open("dataset.p", 'rb'))
X_train = dataset["data"]
y_train = dataset["output"]

# split dataset on train validation (80% training and 20% test)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape[0])
print("Validation set size:", X_validation.shape[0])

adam = Adam(lr=learning_rate)
model.compile(optimizer=adam,
              loss='mse')

# in samples_per_epoch I multiply by 2 because I augment that dataset
history = model.fit_generator(generator=generator(X_train, y_train, batch_size=batch_size),
                              validation_data=generator(X_validation, y_validation, augment_data=False),
                              nb_epoch=epochs, samples_per_epoch=X_train.shape[0], nb_val_samples=X_validation.shape[0])

model.save('model.h5')
