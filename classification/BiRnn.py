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
    # print(len(onehot_encoded))
    return onehot_encoded


target = one_hot_encoder(target)

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=10)

learning_rate = 0.001
# max_samples = 100000
training_iters = 1000
display_size = 10
batch_size = len(data_train)

# 实际上图的像素列数，每一行作为一个输入，输入到网络中。
n_input = 13
# LSTM cell的展开宽度，对于图像来说，也是图像的行数
# 也就是图像按时间步展开是按照行来展开的。
n_step = 26
# LSTM cell个数
n_hidden = 256
n_class = 4
# dropout_rate = tf.placeholder("float", name='Drop_out_keep_prob')
dropout_rate = 0.5

x = tf.placeholder(tf.float32, shape=[None, n_step, n_input], name="inputs")
y = tf.placeholder(tf.float32, shape=[None, n_class], name="outputs")

# 这里的参数只是最后的全连接层的参数，调用BasicLSTMCell这个op，参数已经包在内部了，不需要再定义。
# Weight = tf.Variable(tf.svd([2 * n_hidden, n_class]))
Weight = tf.Variable(tf.random_normal([2 * n_hidden, n_class]))  # 参数共享力度比cnn还大
# Weight = tf.get_variable("Weight", shape=([2 * n_hidden, n_class]), initializer=tf.contrib.layers.xavier_initializer())
bias = tf.Variable(tf.random_normal([n_class]))


# class BiLSTM(is_learning)

def BiRNN(x, weights, biases):
    with tf.variable_scope('BiRNN', initializer=tf.orthogonal_initializer()):
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # 把转置后的矩阵reshape成n_input列，行数不固定的矩阵。
        # 对一个batch的数据来说，实际上有bacth_size*n_step行。
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])  # -1,表示样本数量不固定
        # 拆分成n_step组
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, n_step, 0)
        # print(x)
        # 调用现成的BasicLSTMCell，建立两条完全一样，又独立的LSTM结构
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.relu)
        # lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
        #     lstm_fw_cell, output_keep_prob=(1 - dropout_rate))
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.relu)
        # lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
        #     lstm_bw_cell, output_keep_prob=(1 - dropout_rate))
        # lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell])
        # lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell])
        # 两个完全一样的LSTM结构输入到static_bidrectional_rnn中，由这个op来管理双向计算过程。
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
        # 最后来一个全连接层分类预测
    return tf.matmul(outputs[-1], weights) + biases


pred = BiRNN(x, Weight, bias)
# 计算损失、优化、精度
tv = tf.trainable_variables()  # 得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
# regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# run图过程。
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step < training_iters:
        batch_x = data_train
        batch_y = target_train
        batch_x = batch_x.reshape((batch_size, n_step, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_size == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print(
                'Iter' + str(step) + ',Loss= %.6f' % loss + ', Train Accurancy= %.5f' % acc)
            # if loss < 0.15:
            #     break
        step += 1
    print("Optimizer Finished!")

    print('accuracy:',
          sess.run(accuracy, feed_dict={x: data_test.reshape([-1, n_step, n_input]), y: target_test}))
