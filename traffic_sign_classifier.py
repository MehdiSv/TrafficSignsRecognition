# Load the modules
import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Reload the data
#pickle_file = 'notMNIST.pickle'
#with open(pickle_file, 'rb') as f:
#  pickle_data = pickle.load(f)
#  train_features = pickle_data['train_dataset']
#  train_labels = pickle_data['train_labels']
#  valid_features = pickle_data['valid_dataset']
#  valid_labels = pickle_data['valid_labels']
#  test_features = pickle_data['test_dataset']
#  test_labels = pickle_data['test_labels']
#  del pickle_data  # Free up memory


print('Data and modules loaded.')

# Load pickled data
#import pickle

training_file = './train.p'
testing_file = './test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
train_features, train_labels = train['features'], train['labels']
test_features, test_labels = test['features'], test['labels']

n_train = len(train_features)

n_test = len(test_features)

# TODO: what's the shape of an image?
image_shape = str(len(train_features[0])) + "x" + str(len(train_features[0]))

n_classes = 43

print("Number of training examples =", n_train)
print("Number of training labels =", len(train_labels))
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

from sklearn.preprocessing import LabelBinarizer
import numpy as np
import tensorflow as tf

def normalize(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    return (image_data * 0.8) / 255. + 0.1

train_features = normalize(train_features)
print('Data normalized')

# Turn labels into numbers and apply One-Hot Encoding
encoder = LabelBinarizer()
encoder.fit(train_labels)
train_labels = encoder.transform(train_labels)
test_labels = encoder.transform(test_labels)

# Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
train_labels = train_labels.astype(np.float32)
train_features = train_features.astype(np.float32)
test_labels = test_labels.astype(np.float32)

print('Labels One-Hot Encoded')

from sklearn.model_selection import train_test_split

# Get randomized datasets for training and validation
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.1,
    random_state=832289)

### Generate data additional (if you want to!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

import pandas as pd
import seaborn as sns

sns.set(style="white", color_codes=True)
train_data = pd.read_pickle('./train.p')

print(train_data['labels'])

features_count = 3072
labels_count = 43
n_hidden_layer = 128 # layer number of features
n_hidden_layer_2 = 128 # layer number of features

k_output = 12

features = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
labels = tf.placeholder(tf.float32)

#features_flat = tf.reshape(features, [-1, features_count])

#weights = tf.Variable(tf.random_normal([features_count, labels_count]))
#biases = tf.Variable(tf.zeros(labels_count))

with tf.variable_scope("foo", reuse=None):
    weights_out = tf.get_variable("weights_outss", shape=[n_hidden_layer_2, labels_count],
           initializer=tf.contrib.layers.xavier_initializer())
    weights_first_layer = tf.get_variable("weights_first_layerss", shape=[32 * 32 * k_output, n_hidden_layer],
           initializer=tf.contrib.layers.xavier_initializer())
    weights_second_layer = tf.get_variable("weights_second_layerss", shape=[n_hidden_layer, n_hidden_layer_2],
           initializer=tf.contrib.layers.xavier_initializer())

    bias_out = tf.get_variable("bias_outss", shape=[labels_count],
           initializer=tf.contrib.layers.xavier_initializer())
    bias_first_layer = tf.get_variable("bias_first_layerss", shape=[n_hidden_layer],
           initializer=tf.contrib.layers.xavier_initializer())
    bias_second_layer = tf.get_variable("bias_second_layerss", shape=[n_hidden_layer_2],
           initializer=tf.contrib.layers.xavier_initializer())
    
    weights_conv = tf.get_variable("weights_conv", shape=[3, 3, 3, k_output],
           initializer=tf.contrib.layers.xavier_initializer())
    bias_conv = tf.get_variable("bias_conv", shape=[k_output],
           initializer=tf.contrib.layers.xavier_initializer())
    
#weights = tf.Variable(tf.truncated_normal((features_count, labels_count)))
#biases = tf.Variable(tf.zeros(labels_count))


weights = {
    'weight_conv': weights_conv,
    'hidden_layer': weights_first_layer,
    'hidden_layer_2': weights_second_layer,
    'out': weights_out
}
biases = {
    'bias_conv': bias_conv,
    'hidden_layer': bias_first_layer,
    'hidden_layer_2': bias_second_layer,
    'out': bias_out
}

### DON'T MODIFY ANYTHING BELOW ###
#Test Cases
from tensorflow.python.ops.variables import Variable


# Feed dicts for training, validation, and test session
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
#test_feed_dict = {features: test_features, labels: test_labels}

conv_layer = tf.nn.conv2d(features, weights['weight_conv'], strides=[1, 1, 1, 1], padding='SAME')
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, biases['bias_conv'])
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)

#conv_layer = tf.reshape(conv_layer, [-1, 32*32*3*k_output])

layer_1 = tf.add(tf.matmul(tf.contrib.layers.flatten(conv_layer), weights['hidden_layer']), biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)
layer_1 = tf.add(tf.matmul(layer_1, weights['hidden_layer_2']), biases['hidden_layer_2'])
layer_1 = tf.nn.relu(layer_1)
logits = tf.matmul(layer_1, weights['out']) + biases['out']

prediction = tf.nn.softmax(logits)
cross_entropy = -tf.reduce_sum(labels * tf.log(prediction + 1e-6), reduction_indices=1)

#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)

# Training loss
loss = tf.reduce_mean(cross_entropy)

#Gradient
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

# Create an operation that initializes all variables
init = tf.initialize_all_variables()

# Test Cases
with tf.Session() as session:
    session.run(init)
    session.run(loss, feed_dict=train_feed_dict)
    session.run(loss, feed_dict=valid_feed_dict)
    biases_data = session.run(biases)

print('Tests Passed!')

# Determine if the predictions are correct
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

print('Accuracy function created.')

import math
from tqdm import tqdm
import matplotlib.pyplot as plt

epochs = 3
batch_size = 50

### DON'T MODIFY ANYTHING BELOW ###
# Gradient Descent


# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = batch_size
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(train_features)/batch_size))

    for epoch_i in range(epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, loss],
                feed_dict={features: batch_features, labels: batch_labels})

            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)

        # Check accuracy against Validation data
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

#loss_plot = plt.subplot(211)
#loss_plot.set_title('Loss')
#loss_plot.plot(batches, loss_batch, 'g')
#loss_plot.set_xlim([batches[0], batches[-1]])
#acc_plot = plt.subplot(212)
#acc_plot.set_title('Accuracy')
#acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
#acc_plot.plot(batches, valid_acc_batch, 'b', label='Validation Accuracy')
#acc_plot.set_ylim([0, 1.0])
#acc_plot.set_xlim([batches[0], batches[-1]])
#acc_plot.legend(loc=4)
#plt.tight_layout()
#plt.show()

print('Validation accuracy at {}'.format(validation_accuracy))

