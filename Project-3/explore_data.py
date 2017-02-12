import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

dataset = pickle.load(open("dataset.p", 'rb'))
X_train = dataset["data"]
y_train = dataset["output"]

img = cv2.imread(X_train[0], 1)
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (160, 80), interpolation=cv2.INTER_LANCZOS4)
#destination = np.zeros(img.shape)
#norm_image = cv2.normalize(img, destination, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#print(norm_image)
plt.imshow(img)
plt.show()