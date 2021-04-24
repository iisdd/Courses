'''
目录:
     1.分割数据集
     2.优化梯度下降算法：
     2.1 不使用任何优化算法
     2.2 mini-batch梯度下降法
     2.3 使用具有动量的梯度下降算法
     2.4 使用Adam算法
'''
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

import opt_utils #
import testCase

# 设置画图基本大小
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


########################### 1.最基本的梯度下降 ###########################
def update_parameters_with_gd(parameters , grads , learning_rate):
    L = len(parameters) // 2

    # 挨个更新参数
    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return parameters

# print('测试update_parameters_with_gd')
# parameters , grads , learning_rate = testCase.update_parameters_with_gd_test_case()
# parameters = update_parameters_with_gd(parameters,grads,learning_rate)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

########################### 1.最基本的梯度下降 ###########################


########################### 2.mini-batch梯度下降 ###########################

# mini_batch训练也要把数据集遍历一遍吧
def random_mini_batches(X , Y , mini_batch_size = 64 , seed = 0):
    # 返回一个列表,里面每个元素为: (mini_batch_X , mini_batch_Y)
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = [] # 最后返回的列表

    # 1.洗牌先
    permutation = list(np.random.permutation(m)) # permutation:全排列
    # 这一步相当于生成了坐标
    shuffled_X = X[ : ,permutation]
    shuffled_Y = Y[ : ,permutation].reshape((1 , m))

    """
    #博主注：
    #如果你不好理解的话请看一下下面的伪代码，看看X和Y是如何根据permutation来打乱顺序的。
    x = np.array([[1,2,3,4,5,6,7,8,9],
				  [9,8,7,6,5,4,3,2,1]])
    y = np.array([[1,0,1,0,1,0,1,0,1]])

    random_mini_batches(x,y)
    permutation= [7, 2, 1, 4, 8, 6, 3, 0, 5]
    shuffled_X= [[8 3 2 5 9 7 4 1 6]
                 [2 7 8 5 1 3 6 9 4]]
    shuffled_Y= [[0 1 0 1 1 1 0 1 0]]
    """

    # 2.分割数据集
    num_complete_minibatches = math.floor(m / mini_batch_size) # floor:向下取整
    for i in range(0 , num_complete_minibatches):
        mini_batch_X = shuffled_X[ : ,i * mini_batch_size : (i+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[ : ,i * mini_batch_size : (i+1) * mini_batch_size]
        mini_batch = (mini_batch_X , mini_batch_Y)
        mini_batches.append(mini_batch)

    # 如果训练集的大小不是mini_batch_size的整数倍,把剩下的也包进来
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[ : ,mini_batch_size * num_complete_minibatches : ]
        mini_batch_Y = shuffled_Y[ : ,mini_batch_size * num_complete_minibatches : ]
        mini_batch = (mini_batch_X , mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# print('测试random_mini_batches')
# X_assess,Y_assess,mini_batch_size = testCase.random_mini_batches_test_case() # m = 148
# mini_batches = random_mini_batches(X_assess,Y_assess,mini_batch_size)
#
# print("第1个mini_batch_X 的维度为：",mini_batches[0][0].shape)
# print("第1个mini_batch_Y 的维度为：",mini_batches[0][1].shape)
# print("第2个mini_batch_X 的维度为：",mini_batches[1][0].shape)
# print("第2个mini_batch_Y 的维度为：",mini_batches[1][1].shape)
# print("第3个mini_batch_X 的维度为：",mini_batches[2][0].shape) # (12288 , 20)
# print("第3个mini_batch_Y 的维度为：",mini_batches[2][1].shape) # (1 , 20)

########################### 2.mini-batch梯度下降 ###########################

########################### 3.包含动量的梯度下降 ###########################
def initialize_velocity(parameters):
    # 初始化速度velocity,零初始化即可,不用偏差修正
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v['dW' + str(l + 1)] = np.zeros_like(parameters['W' + str(l+1)])
        v['db' + str(l + 1)] = np.zeros_like(parameters['b' + str(l+1)])

    return v

# print('测试initialize_velocity')
# parameters = testCase.initialize_velocity_test_case()
# v = initialize_velocity(parameters)
#
# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))


def update_parameters_with_momentum(parameters , grads , v , beta , learning_rate):
    # 更新v与更新参数
    L = len(parameters) // 2
    for l in range(L):
        # 更新v,其中beta -> 1.
        v['dW' + str(l + 1)] = beta * v['dW' + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta * v['db' + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]

        # 更新参数
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * v['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v['db' + str(l + 1)]

    return parameters , v

# print('测试update_parameters_with_momentum')
# parameters,grads,v = testCase.update_parameters_with_momentum_test_case()
# update_parameters_with_momentum(parameters,grads,v,beta=0.9,learning_rate=0.01)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))

########################### 3.包含动量的梯度下降 ###########################

########################### 4.Adam梯度下降 ###########################
def initialize_adam(parameters):
    # 初始化v和s
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v['dW' + str(l + 1)] = np.zeros_like(parameters['W' + str(l + 1)])
        v['db' + str(l + 1)] = np.zeros_like(parameters['b' + str(l + 1)])

        s['dW' + str(l + 1)] = np.zeros_like(parameters['W' + str(l + 1)])
        s['db' + str(l + 1)] = np.zeros_like(parameters['b' + str(l + 1)])

    return (v , s)

# print('测试initialize_adam')
# parameters = testCase.initialize_adam_test_case()
# v , s = initialize_adam(parameters)
#
# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))
# print('s["dW1"] = ' + str(s["dW1"]))
# print('s["db1"] = ' + str(s["db1"]))
# print('s["dW2"] = ' + str(s["dW2"]))
# print('s["db2"] = ' + str(s["db2"]))

def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate=0.01,beta1=0.9,
                                beta2=0.999,epsilon=1e-8):
    # t是迭代次数,epsilon是防止除以0的小数字
    # v是之前梯度的指数加权平均,s是之前梯度的平方的指数加权平均
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):

        # v更新
        v['dW' + str(l + 1)] = beta1 * v['dW' + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        # v修正
        v_corrected['dW' + str(l + 1)] = v['dW' + str(l + 1)] / (1 - beta1 ** t)
        v_corrected['db' + str(l + 1)] = v['db' + str(l + 1)] / (1 - beta1 ** t)

        # s更新
        s['dW' + str(l + 1)] = beta2 * s['dW' + str(l + 1)] + (1 - beta2) * (grads['dW' + str(l + 1)] ** 2)
        s['db' + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * (grads['db' + str(l + 1)] ** 2)

        # s修正
        s_corrected['dW' + str(l + 1)] = s['dW' + str(l + 1)] / (1 - beta2 ** t)
        s_corrected['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - beta2 ** t)

        # 更新参数
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * \
            (v_corrected['dW' + str(l + 1)]) / (np.sqrt(s_corrected['dW' + str(l + 1)] + epsilon))
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * \
            (v_corrected['db' + str(l + 1)]) / (np.sqrt(s_corrected['db' + str(l + 1)] + epsilon))

    return (parameters , v , s)

# print('测试update_parameters_with_adam')
# parameters , grads , v , s = testCase.update_parameters_with_adam_test_case()
# update_parameters_with_adam(parameters,grads,v,s,t=2)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))
# print('s["dW1"] = ' + str(s["dW1"]))
# print('s["db1"] = ' + str(s["db1"]))
# print('s["dW2"] = ' + str(s["dW2"]))
# print('s["db2"] = ' + str(s["db2"]))

########################### 4.Adam梯度下降 ###########################


############################## 实例测试 ##############################

# 定义模型
def model(X , Y , layers_dims , optimizer , learning_rate = 0.0007 ,
          mini_batch_size = 64 , beta = 0.9 , beta1 = 0.9 , beta2 = 0.999 ,
          epsilon = 1e-8 , num_epochs = 10000 , print_cost = True , is_plot = True):
    # layers_dims : 包含每层节点数的列表
    L = len(layers_dims)
    costs = []
    t = 0  # 学习过的minibatch的计数器,用于偏差修正
    seed = 10

    # 初始化参数
    parameters = opt_utils.initialize_parameters(layers_dims)

    # 选择优化器
    if optimizer == 'gd':
        pass  # 直接梯度下降就行,不需要额外初始化
    elif optimizer == 'momentum':
        v = initialize_velocity(parameters)
    elif optimizer == 'adam':
        v , s = initialize_adam(parameters)
    else:
        print('optimizer参数错误,程序退出')
        exit(1)

    # 开始train
    for i in range(num_epochs):
        # 每次训练玩一轮都要重新编排minibatch
        seed += 1  # 重新随机排序
        minibatches = random_mini_batches(X , Y , mini_batch_size , seed)

        for minibatch in minibatches:
            # 一个一个minibatch来
            (minibatch_X , minibatch_Y) = minibatch

            # 前向传播
            A3 , cache = opt_utils.forward_propagation(minibatch_X , parameters)

            # 计算误差
            cost = opt_utils.compute_cost(A3 , minibatch_Y)

            # 反向传播
            grads = opt_utils.backward_propagation(minibatch_X , minibatch_Y , cache)

            # 更新参数
            if optimizer == 'gd':
                parameters = update_parameters_with_gd(parameters , grads , learning_rate)
            elif optimizer == 'momentum':
                parameters , v = update_parameters_with_momentum(parameters , grads , v , beta , learning_rate)
            elif optimizer == 'adam':
                t += 1
                parameters , v , s = update_parameters_with_adam(parameters , grads , v , s , t , learning_rate , beta1 , beta2 , epsilon)

        if i % 100 == 0: # 记录误差值
            costs.append(cost) # 用来画图
            if print_cost and i % 1000 == 0:
                print("第" + str(i) + "次遍历整个数据集，当前误差值：" + str(cost))

    # 训练完是否绘制曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

    return parameters



train_X , train_Y = opt_utils.load_dataset(is_plot = False) # 月亮型的数据集
n_x = train_X.shape[0]
# 1.普通的梯度下降
print('\n普通的梯度下降:')
layers_dims = [n_x , 5 , 2 , 1]
parameters = model(train_X , train_Y , layers_dims , optimizer='gd' , is_plot=True )

#预测
preditions = opt_utils.predict(train_X,train_Y,parameters)

#绘制分类图
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)


# 2.带动量的梯度下降
print('\n带动量的梯度下降:')
layers_dims = [n_x,5,2,1]
#使用动量的梯度下降
parameters = model(train_X, train_Y, layers_dims, beta=0.9,optimizer="momentum",is_plot=True,)

#预测
preditions = opt_utils.predict(train_X,train_Y,parameters)

#绘制分类图
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
# 没做偏差修正,和普通梯度下降没啥区别


# 3.adam梯度下降
print('\nadam梯度下降:')
layers_dims = [n_x, 5, 2, 1]
#使用Adam优化的梯度下降
parameters = model(train_X, train_Y, layers_dims, optimizer="adam",is_plot=True,)

#预测
preditions = opt_utils.predict(train_X,train_Y,parameters)

#绘制分类图
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)

############################## 实例测试 ##############################