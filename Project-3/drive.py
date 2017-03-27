import argparse
import base64
from datetime import datetime
import os
import shutil
import cv2

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.utils.visualize_util import plot

from keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
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
    img = np.expand_dims(img, axis=2)
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = np.float(data["steering_angle"])
        # The current throttle of the car
        throttle = np.float(data["throttle"])
        # The current speed of the car
        speed = np.float(data["speed"])
        # The current image from the center camera of the car
        imgString = data["image"]
        original_image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(original_image)
        image = process_image(image_array)
        output = model.predict([image[None, :, :, :], np.array([[steering_angle, throttle, speed]])], batch_size=1)
        #plot(model, show_layer_names=False, show_shapes=True, to_file="model1.png")
        steering_angle = output[0][0]
        throttle = output[0][1] * 1.4
        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            original_image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
