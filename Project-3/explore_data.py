import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

brightness = 0

dataset = pickle.load(open("dataset.p", 'rb'))
X_train = dataset["data"]
y_train = dataset["output"]

# histogram steering angle
steering_angles = y_train[:, 0]
plt.hist(steering_angles, bins=11)
plt.title("Steering angle histogram")
plt.xlabel("Steering angle")
plt.ylabel("Counts")
plt.show()

img = cv2.imread(X_train[np.random.randint(X_train.shape[0])][0], 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
original_image = np.copy(img)
flipped_image = cv2.flip(img, 1)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(original_image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(flipped_image)
ax2.set_title('Flipped image', fontsize=40)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

original_image = np.copy(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(original_image)
ax1.set_title('Cropped Image', fontsize=40)

ax2.imshow(img, cmap='gray')
ax2.set_title('Grayscale image', fontsize=40)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

original_image = np.copy(img)
img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(original_image, cmap='gray')
ax1.set_title('Grayscaled Image', fontsize=40)

ax2.imshow(img, cmap='gray')
ax2.set_title('Resized image', fontsize=40)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

original_image = np.copy(img)
brightness = 30
img = np.where((255 - img) < brightness, 255, img + brightness)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(original_image, cmap='gray')
ax1.set_title('Resized Image', fontsize=40)

ax2.imshow(img, cmap='gray')
ax2.set_title('Brightness augmented image', fontsize=40)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
