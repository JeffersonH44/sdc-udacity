import cv2
import pandas as pd
import pickle

record_folder = "./records/"
correction = 0.2

csv = pd.read_csv(record_folder + "driving_log.csv", header=None)

center_image = pd.concat([csv[0], csv[1], csv[2]])
steering_angle = pd.concat([csv[3], csv[3] + correction, csv[3] - correction])
print(center_image.shape)

dataset = {"data": center_image.as_matrix(), "output": steering_angle.as_matrix()}
pickle.dump(dataset, open("dataset.p", "wb"))




