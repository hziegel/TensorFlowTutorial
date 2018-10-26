import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

dataImage = tf.placeholder("float", [None, 784])
digitsRecognition = tf.placeholder("float", [None,10])

weights = tf.Variable(tf.zeros([784,10]))
bias = tf.Variable(tf.zeros([10]))

with tf.name_scope("wx_b") as scope:
    model = tf.nn.softmax(tf.matmul(dataImage, weights) + bias)

weightSummary = tf.histogram_summary("weights", weights)
biasSummary = tf.histogram_summary("biases", bias)
