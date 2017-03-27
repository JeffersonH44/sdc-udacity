import cv2
import pandas as pd
import pickle

record_folder = "./records/"
correction = 0.05

csv = pd.read_csv(record_folder + "driving_log.csv", header=None) # data from track 1

columns = [0, 1, 2, 3]

val1 = csv[[0, 3, 4, 6]]
val1.columns = columns
val2 = csv[[1, 3, 4, 6]]
val2.columns = columns
val3 = csv[[2, 3, 4, 6]]
val3.columns = columns

center_image = pd.concat([val1, val2, val3], ignore_index=True) # concat all images from both sources (center, left and right)

val1 = csv[[3, 4]]
val2 = csv[[3, 4]]
val2[3] += correction
val3 = csv[[3, 4]]
val3[3] -= correction
steering_angle = pd.concat([val1, val2, val3]) # predictions of each image and correction added for the right and left cameras

dataset = {"data": center_image.as_matrix(), "output": steering_angle.as_matrix()}
pickle.dump(dataset, open("dataset.p", "wb"))




