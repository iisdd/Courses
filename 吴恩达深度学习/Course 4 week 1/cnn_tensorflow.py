# 加一个模型保存
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops

import cnn_utils
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = cnn_utils.load_dataset()
np.random.seed(1)

# index = 6
# plt.imshow(X_train_orig[index])
# plt.show()
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

# 数据处理,变成one-hot模式
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T
# print ("number of training examples = " + str(X_train.shape[0]))
# print ("number of test examples = " + str(X_test.shape[0]))
# print ("X_train shape: " + str(X_train.shape))
# print ("Y_train shape: " + str(Y_train.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

# 样本数量不固定,写出None
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    # tf.placeholder(数据类型,[shape])
    X = tf.placeholder(tf.float32,[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32,[None, n_y])
    return X,Y

# X , Y = create_placeholders(64,64,3,6)
# print ("X = " + str(X))
# print ("Y = " + str(Y))

# 初始化参数 w.shape = [f,f,n_c_prev,n_c_next]
def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable('W1', [4, 4, 3, 8] , initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {
        'W1':W1,
        'W2':W2,
    }
    return parameters

# # 测试
# tf.reset_default_graph()
# with tf.Session() as sess_test:
#     parameters = initialize_parameters()
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#     print("W1 = " + str(parameters["W1"].eval()[1, 1, 1]))
#     print("W2 = " + str(parameters["W2"].eval()[1, 1, 1]))
#
#     sess_test.close()

def forward_propagation(X, parameters):
    # 前向传播
    W1 = parameters['W1']
    W2 = parameters['W2']
    # 结构: 卷积->激活->池化->卷积->激活->池化->全连接
    # conv2d: stride:1 , 填充方式: 'same'
    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME') # X:输入, W1:filter, strides: 分别对于(m, n_H_prev, n_W_prev, n_C_prev)的滑动步长
    A1 = tf.nn.relu(Z1)
    # 池化: 窗口为8x8, 步长也是8x8
    P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
    # 压平送入FC
    P = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P , 6 , activation_fn = None) # 没有激活函数,线性输出
    return Z3

# # 测试
# tf.reset_default_graph()
# np.random.seed(1)
#
# with tf.Session() as sess_test:
#     X, Y = create_placeholders(64, 64, 3, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#
#     a = sess_test.run(Z3, {X: np.random.randn(2, 64, 64, 3), Y: np.random.randn(2, 6)})
#     print("Z3 = " + str(a))
#
#     sess_test.close()

# 计算cost
def compute_cost(Z3 , Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3 , labels=Y))
    return cost


# # 测试
# tf.reset_default_graph()
#
# with tf.Session() as sess_test:
#     np.random.seed(1)
#     X, Y = create_placeholders(64, 64, 3, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#     a = sess_test.run(cost, {X: np.random.randn(4, 64, 64, 3), Y: np.random.randn(4, 6)})
#     print("cost = " + str(a))
#
#     sess_test.close()

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001,
          num_epochs = 10, minibatch_size = 64, print_cost = True, isPlot = True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    # 创建ph
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    # 初始化参数
    parameters = initialize_parameters()
    # 前向传播
    Z3 = forward_propagation(X, parameters)
    # 计算成本
    cost = compute_cost(Z3, Y)
    # 选择优化器反向传播
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # 加一个模型保存
    saver = tf.train.Saver()
    # 全局初始化
    init = tf.global_variables_initializer()
    # 开始训练
    with tf.Session() as sess:
        # 初始化参数
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0                          # 记录这一次遍历样本的平均误差
            num_minibatches = int(m / minibatch_size)   # 整体样本分成了几个minibatch
            seed += 1
            minibatches = cnn_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed) # 返回一个列表,里面是分出来的minibatch
            # 对每个minibatch进行一次训练
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                # 训练&计算cost
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                # 累加成本值
                minibatch_cost += temp_cost/num_minibatches

            if print_cost:
                if epoch % 5 == 0:
                    print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost))

            # 记录总成本走势
            costs.append(minibatch_cost)

        # 训练完了画图
        if isPlot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # 开始预测数据
        # 计算当前的预测情况
        predict_op = tf.arg_max(Z3, 1)
        corrent_prediction = tf.equal(predict_op, tf.arg_max(Y, 1))

        # 计算准确度
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction, "float"))
        print("corrent_prediction accuracy= " + str(accuracy))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuary = accuracy.eval({X: X_test, Y: Y_test})

        print("训练集准确度：" + str(train_accuracy)) # 0.9814815
        print("测试集准确度：" + str(test_accuary))   # 0.85833335

        return (train_accuracy, test_accuary, parameters)

# 启动
_, _, parameters = model(X_train, Y_train, X_test, Y_test,num_epochs=500)