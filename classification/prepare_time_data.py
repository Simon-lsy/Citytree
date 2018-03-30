# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from pandas import DataFrame
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time

df = pd.read_csv('citytree_type_1990_to_2015.csv')
# print(len(df))
# print(list(df))

new_df = df[(df.type > 0) & (df.type < 8) & (df.type != 5)].copy()
print(len(new_df))

data = new_df.iloc[:, 5:].values
target = new_df.iloc[:, 3].values
print(target)


def one_hot_encoder(list):
    values = np.array(list)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    # binary encode(one-hot-encode)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


target = one_hot_encoder(target)

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=0)

# N, data_len = data_train.shape
# ind_N = np.random.choice(N, batch_size, replace=False)
# X_batch = data_train[ind_N]

# hyperparameters
lr = 0.001  # learning rate
training_iters = 100000  # train step上限
batch_size = len(data_train)
n_inputs = 13  # 13 features
n_steps = 26  # 26 rows -> time stamps from 1990 to 2015
n_hidden_units = 256  # hidden units
n_classes = 4  # type of city

# tf input
xs = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="inputs")
ys = tf.placeholder(tf.float32, [None, n_classes], name="outputs")
# W & b
weights = {
    'in': tf.Variable(tf.random_uniform([n_inputs, n_hidden_units], -1.0, 1.0), name="in_w"),
    'out': tf.Variable(tf.random_uniform([n_hidden_units, n_classes], -1.0, 1.0), name="out_w"),
}
b = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units]), name="in_bias"),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), name="out_bias"),
}


def RNN(X, weights, bias):
    # hidden_layer for input
    # X : (128, 28, 28)
    with tf.name_scope("inlayer"):
        X = tf.reshape(X, [-1, n_inputs])
        X_in = tf.matmul(X, weights['in']) + bias['in']
        X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    # RNN cell
    with tf.name_scope("RNN_CELL"):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units)
        # _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        # ouputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, dtype=tf.float32)
    # out layer
    with tf.name_scope('outlayer'):
        results = tf.matmul(states[1], weights['out']) + bias['out']
    return results


pred = RNN(xs, weights, b)
# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
# accuracy
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# run
init = tf.global_variables_initializer()
epochs = int(training_iters / batch_size)
st = time.time()
with tf.Session() as sess:
    sess.run(init)
    batch = len(target_train) / batch_size
    print('batch:' + str(batch))
    for epoch in range(epochs):
        # N, data_len = data_train.shape
        # ind_N = np.random.choice(N, batch_size, replace=False)
        batch_x = data_train
        batch_y = target_train

        batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_op, feed_dict={xs: batch_x, ys: batch_y})
        # print('Loss = ', sess.run(cost, feed_dict={xs: batch_x, ys: batch_y}))
        if epoch % 10 == 0:
            print('epoch:', epoch + 1, 'accuracy:',
                  sess.run(accuracy, feed_dict={xs: data_test.reshape([-1, n_steps, n_inputs]), ys: target_test}))
    end = time.time()
    print('*' * 30)
    print('training finish.\ncost time:', int(end - st), 'seconds\naccuracy:',
          sess.run(accuracy, feed_dict={xs: data_test.reshape([-1, n_steps, n_inputs]), ys: target_test}))


# class = 4: accuracy:70%+-
# class = 5: accuracy:80%+-
#
# BiRNN:
# class = 4: accuracy:80%+-
# class = 5: accuracy:70%+-