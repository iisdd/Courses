# 例子：带有一个隐藏层的平面数据分类

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary , sigmoid , load_planar_dataset , load_extra_datasets
# planar : 平面的

np.random.seed(1)

X,Y = load_planar_dataset()
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral) # 散点图
# plt.show()                                # 和朵花一样
shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]                              # 训练集数量

print('X的维度为: ' + str(shape_X))
# (2 , 400) 两个坐标轴确定一个点
print('Y的维度为: ' + str(shape_Y))
# (1 , 400) 输出一个颜色的点(分类)
print('数据集里的数据有: ' + str(m) + '个')   # 400个

##################查看简单的 logistic 回归的分类效果##########################
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T,Y.T)
#
# plot_decision_boundary(lambda x: clf.predict(x), X, Y) #绘制决策边界
# plt.title("Logistic Regression") #图标题
# LR_predictions  = clf.predict(X.T) #预测结果
# print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
# 		np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
#        "% " + "(正确标记的数据点所占的百分比)")
# plt.show()
# 只能简单的一条线分开
############################################################################

def layer_sizes(X ,Y): # 确定输入层、隐藏层、输出层的神经元数
    '''
    :param X: 输入数据集
    :param Y: 输出标签
    :return: n_x : 输入层数量 , n_h : 隐藏层数量 , n_y : 输出层数量
    '''
    n_x = X.shape[0]
    n_h = 4 # 隐藏层神经元数
    n_y = Y.shape[0]

    return (n_x , n_h , n_y)

# print('=============测试layer_size=============')
# # testCases里有可以测试的 X , Y
# X_assess_ , Y_assess_ = layer_sizes_test_case()   # X : (5,3) , Y : (2,3)
# (n_x_ , n_h_ , n_y_) = layer_size(X_assess_ , Y_assess_)
# print('输入层的节点数量为: n_x = ' , n_x_)
# print('隐藏层的节点数量为: n_h = ' , n_h_)
# print('输出层的节点数量为: n_y = ' , n_y_)

def initialize_parameters(n_x , n_h , n_y): # 初始化权重参数
    '''
    :param n_x: 输入节点数
    :param n_h: 隐藏层节点数
    :param n_y: 输出层节点数
    :return: W1 shape:(n_h , n_x) , b1 shape:(n_h , 1) , W2 shape:(n_y , n_h) , b2 shape:(n_y , 1)
    '''
    np.random.seed(2)  # 可复现
    # 注意W1、W2不可以一样(破除对称性),b1、b2可以相同都为 0
    W1 = np.random.randn(n_h , n_x) * 0.01
    b1 = np.zeros((n_h , 1))
    W2 = np.random.randn(n_y , n_h) * 0.01
    b2 = np.zeros((n_y , 1))

    # 使用 assert保证 shape
    assert(W1.shape == (n_h , n_x))
    assert(b1.shape == (n_h , 1))
    assert(W2.shape == (n_y , n_h))
    assert(b2.shape == (n_y , 1))

    parameters = {'W1' : W1,
                  'b1' : b1,
                  'W2' : W2,
                  'b2' : b2}
    return parameters

# print('================测试initialize_parameters===============')  # 还是用testCases里的例子
# n_x_ , n_h_ , n_y_ = initialize_parameters_test_case()  # (2,4,1)
# parameters_ = initialize_parameters(n_x_ , n_h_ , n_y_)
# print('W1 = ' + str(parameters_['W1']))
# print('b1 = ' + str(parameters_['b1']))
# print('W2 = ' + str(parameters_['W2']))
# print('b2 = ' + str(parameters_['b2']))

def forward_propagation(X , parameters): # 前向传播
    '''
    :param X: 输入数据
    :param parameters: W1 , b1 , W2 , b2
    :return: A2:用sigmoid函数激活的输出 , cache:字典 , 包括'Z1' , 'A1' , 'Z2' , 'A2'
    '''
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

# 前向传播计算 A2
    Z1 = np.dot(W1 , X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2 , A1) + b2
    A2 = sigmoid(Z2)  # planar_utils里的

    # 又用断言来保证格式
    assert(A2.shape == (1 , X.shape[1]))  # 输出为(1 , m)
    cache = {
        'Z1' : Z1,
        'A1' : A1,
        'Z2' : Z2,
        'A2' : A2,
    }
    return A2,cache

# # 测试 forward_propagation
# print('===========================测试forward_propagation=======================')
# X_assess_ , parameters_ = forward_propagation_test_case()
# _ , cache_ = forward_propagation(X_assess_ , parameters_)
# print('Z1 :' , cache_['Z1'] , '\nA1 : ' , cache_['A1'] , '\nZ2 : ' , cache_['Z2'] , '\nA2 : ' , cache_['A2'])

def compute_cost(A2 , Y , parameters): # 计算交叉熵成本
    '''
    :param A2: 预测值
    :param Y: 真实值
    :param parameters: 后面可以用于正则化防止过拟合
    :return: 交叉熵成本 cost
    '''
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']

    # 计算成本cost
    logprobs = np.multiply(np.log(A2) , Y) + np.multiply((1 - Y) , np.log(1 - A2))  # np.multiply : 按位乘
    # logprobs越大越好
    cost = -np.sum(logprobs) / m
    # cost越小越好
    cost = float(np.squeeze(cost))  # 把 cost降维然后变成浮点数

    assert(isinstance(cost , float))

    return cost

# # 测试compute_cost
# print('===============测试compute_cost=================')
# A2_ , Y_assess_ , parameters_ = compute_cost_test_case()
# print('cost = ' , compute_cost(A2_ , Y_assess_ , parameters_))

def backward_propagation(parameters , cache , X , Y):# 反向传播部分,利用前向传播时的cache来确定梯度grads
    '''
    :param parameters: W1、b1、W2、b2这些神经网络参数
    :param cache: Z1、A1、Z2、A2这些中间变量
    :param X: 输入数据(2 , m)
    :param Y: 标签(1 , m)
    :return: grads : 包含 dW1、db1、dW2、db2的字典
    '''
    # 6个式子计算 dZ2、dW2、db2、dZ1、dW1、db1
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2 , A1.T)                              # dot:矩阵乘法
    db2 = (1 / m) * np.sum(dZ2 , axis=1 , keepdims=True)            # axis = 1 : 水平求和
    dZ1 = np.multiply(np.dot(W2.T , dZ2) , 1 - np.power(A1 , 2))    # tan求导sec^2 = 1 - tan^2
    dW1 = (1 / m) * np.dot(dZ1 , X.T)
    db1 = (1 / m) * np.sum(dZ1 , axis=1 , keepdims=True)
    grads = {
        'dW1' : dW1,
        'db1' : db1,
        'dW2' : dW2,
        'db2' : db2,
    }
    return grads

# # 测试 backward_propagation
# print('=============测试 backward_propagation===============')
# parameters_ , cache_ , X_assess_ , Y_assess_ = backward_propagation_test_case()
#
# grads_ = backward_propagation(parameters_ , cache_ , X_assess_ , Y_assess_)
# print('dW2 = ' , grads_['dW2'])
# print('db2 = ' , grads_['db2'])
# print('dW1 = ' , grads_['dW1'])
# print('db1 = ' , grads_['db1'])
# db行数和 dW相同,列数永远是 1,dW行数和后一层神经元数相同,列数和前一层神经元数相同

def update_parameters(parameters , grads , learning_rate = 1.2): # 参数更新部分
    '''
    :param parameters: W , b
    :param grads: dW , db
    :param learning_rate: 学习速率
    :return: 更新好的parameters
    '''
    W1 , W2 = parameters['W1'] , parameters['W2']
    b1 , b2 = parameters['b1'] , parameters['b2']

    dW1 , dW2 = grads['dW1'] , grads['dW2']
    db1 , db2 = grads['db1'] , grads['db2']

    # grads梯度是朝着cost变大的方向,所以这里都用 -=
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {
        'W1' : W1,
        'b1' : b1,
        'W2' : W2,
        'b2' : b2,
    }

    return parameters

# # 测试update_parameters
# print('==============测试update_parameters================')
# parameters_ , grads_ = update_parameters_test_case()
# parameters_ = update_parameters(parameters_ , grads_)
# print('W1 = ' , parameters_['W1'])
# print('b1 = ' , parameters_['b1'])
# print('W2 = ' , parameters_['W2'])
# print('b2 = ' , parameters_['b2'])


def nn_model(X , Y , n_h , num_iterations , print_cost = False): # 把各种功能整合到 nn_model()里
    '''
    :param X: 数据集(2 , m)
    :param Y: 标签(1 , m)
    :param n_h: 隐藏层数量
    :param num_iterations: 迭代次数
    :param print_cost: 每迭代 1000次打印一次cost
    :return: parameters: 返回学习后的参数,可以用来预测
    '''
    np.random.seed(3)
    n_x , n_y = layer_sizes(X , Y)[0] , layer_sizes(X , Y)[2]

    parameters = initialize_parameters(n_x , n_h , n_y)
    # W1 = parameters['W1']
    # b1 = parameters['b1']
    # W2 = parameters['W2']
    # b2 = parameters['b2']

    for i in range(num_iterations): # 前向传播 -> 计算cost -> 梯度回传 -> 更新参数
        A2 , cache = forward_propagation(X , parameters)
        cost = compute_cost(A2 , Y , parameters)
        grads = backward_propagation(parameters , cache , X , Y)
        parameters = update_parameters(parameters , grads , learning_rate=0.5)

        if print_cost:
            if i % 1000 == 0:
                print('第%d次迭代 : cost = %.4f' % (i , cost))

    return parameters

# # 测试 nn_model()
# print('============测试nn_model===============')
# X_assess_ , Y_assess_ = nn_model_test_case()
#
# parameters_ = nn_model(X_assess_ , Y_assess_ , 4 , num_iterations=10000 , print_cost=False)
# print('W1 = ' , parameters_['W1'])
# print('b1 = ' , parameters_['b1'])
# print('W2 = ' , parameters_['W2'])
# print('b2 = ' , parameters_['b2'])


def predict(parameters , X): # 预测结果部分(二分类)
    '''
    :param parameters: W & b
    :param X: 输入数据(2 , m)
    :return: predictions : 模型预测的向量(红色 : 0 / 蓝色 : 1)
    '''
    A2 , cache = forward_propagation(X , parameters)
    predictions = np.round(A2)

    return predictions

# # 测试prediction
# print('==============测试prediction===============')
# parameters_ , X_assess_ = predict_test_case()
# predictions_ = predict(parameters_ , X_assess_)
# print('预测值 : ' , predictions_)



# 开始使用,训练10000次
parameters = nn_model(X ,Y , n_h = 4 , num_iterations=10000 , print_cost=True)

# 画图部分,绘制边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

# 打印准确率
predictions = predict(parameters, X)
print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

# 试试不同数量的隐藏层节点数
plt.figure(figsize=(16 , 32))
hidden_layer_sizes = [1 , 2 , 3 , 4 , 5 , 20 , 50]
for i , n_h in enumerate(hidden_layer_sizes):       # for循环的grid search
    plt.subplot(5 , 2 , i+1)
    plt.title('Hidden Layer of size %d'%n_h)
    parameters = nn_model(X , Y , n_h , num_iterations=5000 , print_cost=False)
    plot_decision_boundary(lambda x : predict(parameters , x.T) , X , Y)
    predictions = predict(parameters , X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print('隐藏层神经元数 : {} , 准确率 : {}%'.format(n_h , accuracy))
plt.show()
# 还是 n_h == 4的效果最好