#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Import Minst data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./tmp/data/", one_hot=True)



# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 256
display_step = 1
examples_to_show = 10


# Network Parameters
# 神经元数量的设置

n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 128  # 2nd layer num features


# 设置Mnist图像的大小，28*28
n_input = 784  # MNIST data imput (img shape:28*28)


#输入
X = tf.placeholder("float", [None, n_input])


#权重的设置
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

def xavier_init(fan_in, fan_out, const=1):
    """
    Xavier initialization of network weights.
    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensor
    
    :param fan_in:fan in of the network(n_features)
    :param fan_out:fan out of the network(n_components)
    :param const: multiplication constant
    """
    low = -const * np.sqrt(6.0/(fan_in + fan_out))
    high = const * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)



#Build the encoder
def encoder(x):
    #encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    #Decoder Hidden layer with sigmoid cativation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


#Build the decoder
def decoder(x):
    #Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
    #Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))
    return layer_2


#Construct model
#编码-解码
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)


#Prediction
#序列-序列
y_pred = decoder_op
#Targets (Lables) are the input data.
y_true = X


#cost function & optimizer
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#尝试交叉编译
#cost = tf.reduce_mean(-tf.reduce_sum(y_pred * tf.log(y_true)))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


# Initializing the variables
# init = tf.global_variablers_inintializer()
init = tf.initialize_all_variables()

#launch the graph
with tf.Session() as sess:
    # 初始化
    sess.run(init)


    # 分数量
    total_batch = int(mnist.train.num_examples/batch_size)

    # Training cycle
    for epoch in range(training_epochs):
        # loop over all batches
        for i in range(total_batch):
            # batch_ys 数字
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)


            # Run optimization op (backprop) and cost op(to get loss_value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})


        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d'% (epoch+1),
                  "cost", "{:.9f}".format(c))

    print("optimization Finished!")
    # Applying encode and decode over test set
    encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # encode_decode = sess.run(y_pred, feed_dict=X)

# compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()


