import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

brightness = 0

dataset = pickle.load(open("dataset.p", 'rb'))
X_train = dataset["data"]
y_train = dataset["output"]

img = cv2.imread(X_train[np.random.randint(X_train.shape[0])], 1)
img = img[60:-25, :, :]
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
img = cv2.flip(img, 1)
img = np.where((255 - img) < brightness, 255, img + brightness)
#destination = np.zeros(img.shape)
#norm_image = cv2.normalize(img, destination, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#print(norm_image)
print(img.shape)
plt.imshow(img)
plt.show()