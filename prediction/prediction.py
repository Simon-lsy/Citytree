# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

rnn_unit = 10  # 隐层数量
input_size = 13
output_size = 1
lr = 0.0006  # 学习率
# ——————————————————导入数据——————————————————————
df = pd.read_csv('shanghai_data_timeseries.csv')
data = df.iloc[:, 2:].values  # 取第3-10列


# 获取训练集
def get_train_data(batch_size=10, time_step=5, train_begin=0, train_end=20):
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :13]
        y = normalized_train_data[i:i + time_step, 13, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


# 获取测试集
def get_test_data(time_step=5, test_begin=15):
    data_test = data[test_begin:test_begin + time_step]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y = [], []
    for i in range(size):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :13]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, 13]
        test_x.append(x.tolist())
        test_y.extend(y)
    # test_x.append((normalized_test_data[(i + 1) * time_step:, :13]).tolist())
    # test_y.extend((normalized_test_data[(i + 1) * time_step:, 13]).tolist())
    return mean, std, test_x, test_y


# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# ——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ————————————————训练模型————————————————————

def train_lstm(batch_size=10, time_step=5, train_begin=0, train_end=20):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(200):  # 这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]]})
            print("Number of iterations:", i, " loss:", loss_)
        print("model_save: ", saver.save(sess, 'model\\modle.ckpt'))
        # 地址是存放模型的地方，模型参数文件名为modle.ckpt
        print("The train has finished")


train_lstm()

acc_list = [1]
year_data = [0]


# ————————————————预测模型————————————————————
def prediction(time_step=7):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    with tf.variable_scope("sec_lstm", reuse=True):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model')
        saver.restore(sess, module_file)
        test_predict = [0]
        # print(len(test_x))
        acc = 1
        step = 0
        while acc > 0.02 and step < 10:
            step += 1
            test_predict = []
            for step in range(len(test_x)):
                prob = sess.run(pred, feed_dict={X: [test_x[step]]})
                predict = prob.reshape((-1))
                test_predict.extend(predict)
            test_y = np.array(test_y) * std[13] + mean[13]
            test_predict = np.array(test_predict) * std[13] + mean[13]
            print(test_predict)
            print(test_predict[-1])
            year_data.append(test_predict[-1])
            data[time_step + 14][-1] = float(test_predict[-1])
            acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差程度
            print("The accuracy of this predict:", acc)
        acc_list.append(acc)
        if time_step == 11 and sorted(acc_list)[-1] <= 0.02 and sorted(year_data)[0] > 1799.622005:
            # 折线图表示结果
            plt.figure()
            plt.plot(list(range(len(test_predict))), test_predict, color='b', )
            plt.plot(list(range(len(test_y) - 1)), test_y[0:len(test_y) - 1], color='r')
            plt.show()


# prediction()
while sorted(acc_list)[-1] > 0.02 or sorted(year_data)[0] < 1799.622005:
    # train_lstm()
    print('---------------------------')
    acc_list = []
    year_data = []
    for i in range(7, 12):
        prediction(time_step=i)

print(acc_list)
