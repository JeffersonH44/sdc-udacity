import os
import cv2

images = [img for img in os.listdir() if img.startswith("sign")]

for image in images:
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("scaled-" + image, img)
