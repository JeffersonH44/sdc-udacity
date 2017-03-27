import cv2
import pandas as pd
import pickle

record_folder = "./records/"
correction = 0.05

csv = pd.read_csv(record_folder + "driving_log.csv", header=None) # data from track 1

columns = [0, 1, 2, 3]

val1 = csv[[0, 3, 4, 6]] # center camera data
val1.columns = columns
val2 = csv[[1, 3, 4, 6]] # left camera data
val2.columns = columns
val3 = csv[[2, 3, 4, 6]] # right camera data
val3.columns = columns

center_image = pd.concat([val1, val2, val3], ignore_index=True) # concat all images from both sources (center, left and right)

val1 = csv[[3, 4]] # predictions with center camera
val2 = csv[[3, 4]] # prediction with left camera
val2[3] += correction # left camera correction
val3 = csv[[3, 4]] # prediction with right camera
val3[3] -= correction # right camera correction
steering_angle = pd.concat([val1, val2, val3]) # predictions of each image and correction added for the right and left cameras

dataset = {"data": center_image.as_matrix(), "output": steering_angle.as_matrix()}
pickle.dump(dataset, open("dataset.p", "wb"))




