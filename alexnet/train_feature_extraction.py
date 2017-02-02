import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# Parameters
nb_classes = 43
epochs = 10
batch_size = 128

# TODO: Load traffic signs data.
train_dataset = pickle.load(open("train.p", 'rb'))
print(train_dataset.keys())

train_x, train_y, sizes, coords = train_dataset['features'], train_dataset['labels'], train_dataset['sizes'], train_dataset['coords']

# TODO: Split data into training and validation sets.
train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, random_state=42, test_size=0.2)
print(train_y.size, validation_x.size)

# TODO: Define placeholders and resize operation.

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(..., feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# TODO: Train and evaluate the feature extraction model.
