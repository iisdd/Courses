# tensorflow入门
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time

np.random.seed(1)

# # 测试例子两个
# # 定义变量
# y_hat = tf.constant(36 , name = 'y_hat') # 预测值
# y = tf.constant(39 , name = 'y')
# # 用变量计算
# loss = tf.Variable((y - y_hat)**2 , name = 'loss')
# # 初始化变量
# init = tf.global_variables_initializer()
#
# with tf.Session() as session: # 创建session
#     session.run(init)  # 运行session
#     print(session.run(loss))

# a = tf.constant(2)
# b = tf.constant(10)
# c = tf.multiply(a,b)
#
# print(c) # 这是一个计算图,还没运行它就不会打印20
#
# sess = tf.Session()
# print(sess.run(c))

# # 占位符
# x = tf.placeholder(tf.int64,name="x")
# sess = tf.Session()
# print(sess.run(2 * x,feed_dict={x:3}))
# sess.close()

############################# 1.导入tensorflow ##############################
# 1.1-线性函数
def linear_function(): # 一层前向传递函数
    np.random.seed(1)

    X = np.random.randn(3 , 1)
    W = np.random.randn(4 , 3)
    b = np.random.randn(4 , 1)

    Y = tf.matmul(W , X) + b
    with tf.Session() as sess: # 打开后自动关
        result = sess.run(Y)
    return result
# # 测试一下
# print('result = ' + str(linear_function()))

# 1.2-计算sigmoid
def sigmoid(z):
    # 变量
    x = tf.placeholder(tf.float32 , name = 'x')
    # 计算
    sigmoid = tf.sigmoid(x) # 这还是个计算图,还没运行
    #  会话运行
    with tf.Session() as sess:
        result = sess.run(sigmoid , feed_dict={x : z}) # feed_dict只用填ph就行
    return result
# # 测试一下
# print('sigmoid(0) = ' , sigmoid(0))
# print('sigmoid(10) = ' , sigmoid(10))

# 1.3-计算成本
# tf.nn.sigmoid_cross_entropy_with_logits(logits = ,labels= ) # logits填预测值a[L],labels填标签值

# 1.4-独热编码(0,1编码)
# tf.one_hot(indices= , depth= , axis= )
# indices是原来的标签向量,横着的,depth是分类数,axis=0代表每个类别都是一竖条,即竖条和为1
def one_hot_matrix(labels , C):
    '''
    Args:
        labels: 标签向量
        C: 分类种数

    Returns: 独热矩阵
    '''
    C = tf.constant(C , name = 'C')
    # 计算图
    one_hot_matrix = tf.one_hot(indices=labels , depth=C , axis=0) # 每个竖条代表一个分类
    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix)
    return one_hot
# # 测试一下
# labels = np.array([1,2,3,0,2,1])
# one_hot = one_hot_matrix(labels,C=4)
# print(one_hot)

# 1.5-初始化为0和1
def ones(shape):
    ones = tf.ones(shape) # 计算图
    with tf.Session() as sess:
        ones = sess.run(ones)
    return ones
# # 测试一下
# print('ones = ' + str(ones([4,3,2]))) # 4行3列每个元素[]里包括两个厚度(两个数)

############################# 1.导入tensorflow ##############################


######################### 2.使用tensorflow构建神经网络 ##########################
# 牢记两个步骤 1.创建计算图 2.运行计算图
# 例子: 64×64像素的手势数字识别(0-5)
# 加载数据集
X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = tf_utils.load_dataset()
# print(X_train_orig.shape) # (1080 , 64 , 64 , 3)

# # 图片展示
# index = 11
# plt.imshow(X_train_orig[index])
# plt.show()
# print("Y = " + str(np.squeeze(Y_train_orig[:,index])))

# 对数据进行扁平化处理
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T #每一列就是一个样本
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0],-1).T
# Xshape:(n_features , m)

#归一化数据
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

#转换为独热矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig,6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig,6)
# Yshape:(C , m)

# print("训练集样本数 = " + str(X_train.shape[1]))
# print("测试集样本数 = " + str(X_test.shape[1]))
# print("X_train.shape: " + str(X_train.shape))
# print("Y_train.shape: " + str(Y_train.shape))
# print("X_test.shape: " + str(X_test.shape))
# print("Y_test.shape: " + str(Y_test.shape))

# 2.1-创建ph
def create_placeholders(n_x , n_y):
    X = tf.placeholder(tf.float32 , [n_x , None] , name = 'X') # 本例中n_x = 64*64*3
    Y = tf.placeholder(tf.float32 , [n_y , None] , name = 'Y') # n_y = 6
    return X , Y
# # 测试一下
# X, Y = create_placeholders(12288, 6)
# print("X = " + str(X))
# print("Y = " + str(Y))

# 2.2-初始化参数
# 注: tf.Variable() 每次都在创建新对象，对于get_variable()来说，
# 对于已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，就创建一个新的。

def initialize_parameters():
    tf.set_random_seed(1)

    # n_hidden: [12288 , 25 , 12 , 6]
    # W用xavier初始化,b用零初始化
    W1 = tf.get_variable('W1',[25 , 12288],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1',[25 , 1],initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2',[12 , 25],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2',[12 , 1],initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3',[6 , 12],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3',[6 , 1],initializer=tf.zeros_initializer())

    parameters = {'W1':W1,'b1':b1,'W2':W2,'b2':b2,'W3':W3,'b3':b3}
    return parameters
# # 测试一下
# tf.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形。
#
# with tf.Session() as sess:
#     parameters = initialize_parameters()
#     print("W1 = " + str(parameters["W1"]))
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))
#     # 只有物理空间,还没被赋值

# 2.3-前向传播
def forward_propagation(X , parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.matmul(W1 , X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2 , A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3 , A2) + b3
    # 最后一层线性输出即可,在求cost的时候再加softmax
    return Z3
# # 测试一下
# tf.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形。
# with tf.Session() as sess:
#     X,Y = create_placeholders(12288,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     print("Z3 = " + str(Z3))

# 2.4-计算成本
# 用tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ..., labels = ...))
def compute_cost(Z3 , Y):
    # Yshape:(C , m),转置过来求reduce_mean,reduce:降维\
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    return cost
# # 测试一下
# tf.reset_default_graph()
#
# with tf.Session() as sess:
#     X,Y = create_placeholders(12288,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     cost = compute_cost(Z3,Y)
#     print("cost = " + str(cost))

# 2.5-反向传播与参数更新
# tensorflow里更新很简单,两句话就行了
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
# _ , c = sess.run([optimizer , cost],feed_dict={X:mini_batch_X,Y:mini_batch_Y})

# 2.6-构建模型
def model(X_train , Y_train , X_test , Y_test ,
          learning_rate = 0.0001 , num_epochs = 1500 ,
          minibatch_size = 32 , print_cost = True , is_plot = True):
    """
    实现一个三层的TensorFlow神经网络：LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX

    参数：
        X_train - 训练集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 1080）
        Y_train - 训练集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 1080）
        X_test - 测试集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 120）
        Y_test - 测试集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 120）
        learning_rate - 学习速率
        num_epochs - 整个训练集的遍历次数
        minibatch_size - 每个小批量数据集的大小
        print_cost - 是否打印成本，每100代打印一次
        is_plot - 是否绘制曲线图

    返回：
        parameters - 学习后的参数

    """
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x , m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X , Y = create_placeholders(n_x , n_y)
    parameters = initialize_parameters() # 这里面都是Variables,优化的时候就只改变Variables的值
    Z3 = forward_propagation(X , parameters)
    cost = compute_cost(Z3 , Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    # 开启会话计算
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0 # 每代迭代的成本
            num_minibatches = int(m / minibatch_size)
            seed += 1 # 每次生成的minibatches都不一样
            minibatches = tf_utils.random_mini_batches(X_train , Y_train , minibatch_size , seed)

            for minibatch in minibatches:
                (minibatch_X , minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer , cost] , feed_dict={X:minibatch_X , Y:minibatch_Y})
                epoch_cost += minibatch_cost

            epoch_cost /= num_minibatches
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                if print_cost and epoch % 100 == 0:
                    print('epoch = ' + str(epoch) + '    epoch_cost = ' + str(epoch_cost))

        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # 保存学习后的参数
        parameters = sess.run(parameters)
        print("参数已经保存到session。")

        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(Z3) , tf.argmax(Y)) # 一个0,1矩阵
        accuracy = tf.reduce_mean(tf.cast(correct_prediction , 'float')) # tf.cast:数据类型转换
        print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

# train起来
#开始时间
start_time = time.clock()
#开始训练
parameters = model(X_train, Y_train, X_test, Y_test)
#结束时间
end_time = time.clock()
#计算时差
print("CPU的执行时间 = " + str(end_time - start_time) + " 秒" )
######################### 2.使用tensorflow构建神经网络 ##########################