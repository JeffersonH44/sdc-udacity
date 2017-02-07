from keras.datasets import cifar10
import tensorflow as tf
import cv2
import sklearn
import numpy as np

from tensorflow.contrib.layers import flatten
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

try_test = True

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    # print(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def normalize_grayscale(image_data):
    a = -100.0
    b = 100.0
    grayscale_min = 0
    grayscale_max = 255
    return a + (((image_data - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))


### Define your architecture here.
### Feel free to use as many code cells as needed.
def LeNet(x, activation=tf.tanh, pooling=tf.nn.avg_pool, keep_prob=0.5):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_B = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=(1, 1, 1, 1), padding='VALID') + conv1_B

    # TODO: Activation.
    conv1 = activation(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = pooling(conv1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_B = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=(1, 1, 1, 1), padding='VALID') + conv2_B

    # TODO: Activation.
    conv2 = activation(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = pooling(conv2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flat = flatten(conv2)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 350.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 350), mean=mu, stddev=sigma))
    fc1_B = tf.Variable(tf.zeros(350))
    fc1 = tf.matmul(flat, fc1_W) + fc1_B

    # TODO: Activation.
    fc1 = activation(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(350, 120), mean=mu, stddev=sigma))
    fc2_B = tf.Variable(tf.zeros(120))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_B

    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    # fc2 = fc1

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(120, n_classes), mean=mu, stddev=sigma))
    fc3_B = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_B

    return logits


f = lambda img: normalize_grayscale(grayscale(img))
# f = lambda img: grayscale(img)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape)
# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_test.shape[1:3]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.unique(y_train).shape[0]

X_train = [f(img) for img in X_train]
X_test = [f(img) for img in X_test]

X_train = np.array(X_train)
X_train = np.reshape(X_train, (n_train, 32, 32, 1))

X_test = np.array(X_test)
X_test = np.reshape(X_test, (n_test, 32, 32, 1))

# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(X_train.shape, X_validation.shape)

### Train your model here.
### Feel free to use as many code cells as needed.
EPOCHS = 100
BATCH_SIZE = 250
rate = 0.0001
try_test = True

## placeholders for data
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    if try_test:
        test_accuracy = evaluate(X_test, y_test)
        # print("EPOCH {} ...".format(i+1))
        print("Test Accuracy = {:.3f}".format(test_accuracy))
        print()
    saver.save(sess, './traffic')
    print("Model saved")