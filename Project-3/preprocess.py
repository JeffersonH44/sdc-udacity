import cv2
import pandas as pd
import pickle

record_folder = "./Records/"

csv = pd.read_csv(record_folder + "driving_log.csv", header=None)

center_image = csv[0]
steering_angle = csv[3]

dataset = {"data": center_image.as_matrix(), "output": steering_angle.as_matrix()}
pickle.dump(dataset, open("dataset.p", "wb"))




