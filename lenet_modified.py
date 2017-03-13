import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

######################
## Load data
######################
import pickle

training_file = 'test.p'
validation_file='valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

######################
## Pre-processing
######################
## Converting to Grayscale
X_train = np.mean(X_train, axis=3, keepdims=True, dtype=np.float32)
X_valid = np.mean(X_valid, axis=3, keepdims=True, dtype=np.float32)
X_test  = np.mean(X_test, axis=3, keepdims=True, dtype=np.float32)

## Shuffle
X_train, y_train = shuffle(X_train, y_train)

######################
## Define the model
######################
def LeNet_modified(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.05
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x12.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 12), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(12))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Layer 2: Convolutional. Input = 28x28x12. Output = 24x24x24.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 24), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(24))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 24x24x24. Output = 12x12x24.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 3: Convolutional. Input = 12x12x24. Output = 10x10x32.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 24, 32), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(32))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    
    # Flatten. Input = 10x10x32. Output = 3200.
    fc0   = flatten(conv3)
    
    # Layer 4: Fully Connected. Input = 3200. Output = 1600.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(3200, 1600), mean = mu, stddev = sigma))    
    fc1_b = tf.Variable(tf.zeros(1600))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1)
    
    # Dropout 1
    fc1 = tf.nn.dropout(fc1, keep_prob=0.7)

    # Layer 5: Fully Connected. Input = 1600. Output = 800.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(1600, 800), mean = mu, stddev = sigma))    
    fc2_b  = tf.Variable(tf.zeros(800))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
        
    # Activation.
    fc2    = tf.nn.relu(fc2)
    
    # Dropout 2
    fc2 = tf.nn.dropout(fc2, keep_prob=0.8)

    # Layer 6: Fully Connected. Input = 800. Output = 200.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(800, 200), mean = mu, stddev = sigma))
    #fc3_W = tf.get_variable("fc3", shape=[800, 200], initializer=tf.contrib.layers.xavier_initializer())
    fc3_b  = tf.Variable(tf.zeros(200))
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b
    
    # Activation.
    fc3    = tf.nn.relu(fc3)
    
    # Dropout 3
    fc3 = tf.nn.dropout(fc3, keep_prob=0.9)
    
    # Layer 7: Fully Connected. Input = 200. Output = 100.
    fc4_W  = tf.Variable(tf.truncated_normal(shape=(200, 100), mean = mu, stddev = sigma))    
    fc4_b  = tf.Variable(tf.zeros(100))
    fc4 = tf.matmul(fc3, fc4_W) + fc4_b
    
    # Activation.
    fc4    = tf.nn.relu(fc4)
    
    # Layer 8: Fully Connected. Input = 100. Output = 43.
    fc5_W  = tf.Variable(tf.truncated_normal(shape=(100, 43), mean = mu, stddev = sigma))    
    fc5_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc4, fc5_W) + fc5_b
    
    return logits
    
###########################
## Train / Validate / Test
###########################
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

## Training Pipeline
rate = 0.0004
EPOCHS = 100
BATCH_SIZE = 128

logits = LeNet_modified(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

## Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

## Train
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
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    
## Evaluate the model
with tf.Session() as sess:
    #saver.restore(sess, tf.train.latest_checkpoint('.'))
    saver.restore(sess, './lenet')

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    print("Model saved")
