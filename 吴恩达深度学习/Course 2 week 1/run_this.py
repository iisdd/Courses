'''
1. 初始化参数：
	1.1：使用0来初始化参数。
	1.2：使用随机数来初始化参数。
	1.3：使用抑梯度异常初始化参数（参见视频中的梯度消失和梯度爆炸）。
2. 正则化模型：
	2.1：使用二范数对二分类模型正则化，尝试避免过拟合。
	2.2：使用随机删除节点(dropout)的方法精简模型，同样是为了尝试避免过拟合。
3. 梯度校验：对模型使用梯度校验，检测它是否在梯度下降的过程中出现误差过大的情况。
'''


import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils  # 第一部分,初始化
import reg_utils   # 第二部分,正则化
import gc_utils    # 第三部分,梯度校验

# 设置默认的画图格式
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

####################################### 第一部分：初始化 ######################################
# EX:2D数据分类
# 数据展示
train_X , train_Y , test_X , test_Y = init_utils.load_dataset(is_plot=False)
#plt.show()
# 两个扣着的圆圈

# 总模型
def model(X , Y , learning_rate = 0.01 , num_iterations = 15000 ,
          print_cost = True , initialization = 'he' , is_plot = True):
    '''
    用已有的三层神经网络init_utils来测试不同的初始化方法对准确率的影响
    Args:
        X: 输入数据,shape:(2 , m)
        Y: 标签,shape:(1 , m)
        learning_rate:
        num_iterations:
        print_cost:
        initialization: 选择初始化的方法
        is_plot:

    Returns:
        parameters: 返回更新后的参数
    '''
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0] , 10 , 5 , 1]
    # 输入X的坐标，输出它的分类

    # 选择初始化参数的类型,初始化的具体过程后面定义
    if initialization == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == 'he':
        parameters = initialize_parameters_he(layers_dims)
    else:
        print('错误的初始化参数,即将关机!!!')
        exit

    # train的部分
    for i in range(0 , num_iterations):
        # 前向传播
        a3 , cache = init_utils.forward_propagation(X , parameters)
        # 计算loss
        cost = init_utils.compute_loss(a3 , Y)
        # 反向传播
        grads = init_utils.backward_propagation(X , Y , cache)
        # 更新参数
        parameters = init_utils.update_parameters(parameters , grads , learning_rate)

        # 记录cost,用来画图
        if i % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print('第%d次迭代,成本值为:%f' % (i , cost))

    # 学完画cost图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    # 返回学习完的参数
    return parameters


############################# 1.零初始化 #############################
def initialize_parameters_zeros(layers_dims):
    '''
    Args:
        layers_dims: 包含每一层节点数的列表

    Returns:
        parameters: 包含所有 W,b的字典
    '''
    parameters = {}

    L = len(layers_dims)

    for l in range(1 , L): # 第1个元素是X的维度,不用管
        parameters['W' + str(l)] = np.zeros((layers_dims[l] , layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l] , 1))

        # 使用断言确保数据格式正确
        assert(parameters['W'+str(l)].shape == (layers_dims[l] , layers_dims[l-1]))
        assert(parameters['b'+str(l)].shape == (layers_dims[l] , 1))

    return parameters


# # 测试initialize_parameters_zeros
# parameters = initialize_parameters_zeros([3,2,1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# # 零初始化train结果
# parameters = model(train_X , train_Y , initialization='zeros' , is_plot=True , )
# # 一条直线,没学到东西
#
# print ("训练集:") # 0.5
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
# print ("测试集:") # 0.5
# predictions_test = init_utils.predict(test_X, test_Y, parameters)
#
# print("predictions_train = " + str(predictions_train))
# print("predictions_test = " + str(predictions_test))
#
# plt.title("Model with Zeros initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)
# # 把所有点都归类为0,所以正确率是0.5,说明零初始化使网络无法打破对称性train不起来

############################# 1.零初始化 #############################



############################ 2.随机初始化 #############################
def initialize_parameters_random(layers_dims):
    '''
    Args:
        layers_dims: 每层节点的列表

    Returns:
        parameters: 包含'W'和'b'的字典
    '''
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1 , L):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l] , layers_dims[l-1]) * 10
        # 放大10倍嗷,不能让它有好效果,不乘10正确率就99%了...
        parameters['b'+str(l)] = np.zeros((layers_dims[l] , 1))

        # 使用断言确保数据格式
        assert(parameters['W'+str(l)].shape == (layers_dims[l] , layers_dims[l-1]))
        assert(parameters['b'+str(l)].shape == (layers_dims[l] , 1))

    return parameters

# # 测试initialze_parameters_random
# parameters = initialize_parameters_random([3, 2, 1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# parameters = model(train_X, train_Y, initialization = "random",is_plot=True , )
# print("训练集：") # 0.83
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
# print("测试集：") # 0.86
# predictions_test = init_utils.predict(test_X, test_Y, parameters)
# # 初始化太大了会导致梯度爆炸,后面会train不动
#
# print(predictions_train)
# print(predictions_test)
#
# plt.title("Model with large random initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)



############################ 2.随机初始化 #############################



######################### 3.抑梯度异常初始化 #############################
def initialize_parameters_he(layers_dims):
    '''
    Args:
        layers_dims:

    Returns:
    老规矩了,略
    '''
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1 , L):
        std = np.sqrt(2 / layers_dims[l - 1]) # 高斯分布的标准差缩放倍数
        parameters['W'+str(l)] = np.random.randn(layers_dims[l] , layers_dims[l-1]) * std
        parameters['b'+str(l)] = np.zeros((layers_dims[l] , 1))

        # assert!!!
        assert(parameters['W'+str(l)].shape == (layers_dims[l] , layers_dims[l-1]))
        assert(parameters['b'+str(l)].shape == (layers_dims[l] , 1))

    return parameters

# # 测试initialize_parameters_he
# parameters = initialize_parameters_he([2, 4, 1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
#
# parameters = model(train_X, train_Y, initialization = "he",is_plot=True)
# print("训练集:") # 0.99
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
# print("测试集:") # 0.96
# init_utils.predictions_test = init_utils.predict(test_X, test_Y, parameters)
#
# plt.title("Model with He initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

######################### 3.抑梯度异常初始化 #############################

####################################### 第一部分：初始化 ######################################

####################################### 第二部分：正则化 ######################################
# EX:守门员发球哪个位置更容易被自己人接到
# 看看数据
train_X , train_Y , test_X , test_Y = reg_utils.load_2D_dataset(is_plot=False)
# plt.show()

# 用三种方法对比模型优劣:
# 1.不使用正则化
# 2.使用L2正则化
# 3.使用dropout

# 总模型
def model(X,Y,learning_rate=0.3,num_iterations=30000,print_cost=True,
          is_plot=True,lambd=0,keep_prob=1):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0] , 20 , 3 , 1]

    # 初始化参数,自带的最高级的初始化
    parameters = reg_utils.initialize_parameters(layers_dims)

    # 开始train
    for i in range(0,num_iterations):
        # 前向传播
        # 判断是否随机删除节点
        if keep_prob == 1:  # 不删除节点
            a3 , cache = reg_utils.forward_propagation(X , parameters)
        elif keep_prob < 1: # 删除节点,之后会定义这个前向传播
            a3 , cache = forward_propagation_with_dropout(X , parameters , keep_prob)
        else:
            print('keep_prob参数错误!程序自毁.')
            exit

        # 计算成本
        # 判断要不要加入L2正则化
        if lambd == 0: # 不加正则化
            cost = reg_utils.compute_cost(a3 , Y)
        else:# 后面定义加入正则化的cost计算
            cost = compute_cost_with_regularization(a3 , Y , parameters , lambd)

        # 反向传播
        # 本例中不同时使用L2正则化和dropout
        assert(lambd == 0 or keep_prob == 1)

        # 分情况反向传播
        if(lambd == 0 and keep_prob == 1): # 最纯粹的反向传播
            grads = reg_utils.backward_propagation(X,Y,cache)
        elif lambd != 0: # 正则化
            grads = backward_propagation_with_regularization(X , Y , cache , lambd)
        elif keep_prob < 1: # dropout
            grads = backward_propagation_with_dropout(X , Y , cache , keep_prob)

        # 更新参数
        parameters = reg_utils.update_parameters(parameters , grads , learning_rate)

        # 记录&打印成本
        if i % 1000 == 0:
            costs.append(cost)
            if (print_cost and i % 10000 == 0):
                print('第%d次迭代,成本值为%f' % (i , cost))


    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


################################## 1.不使用正则化 ####################################
# parameters = model(train_X , train_Y , is_plot=True)
# print('训练集: ') # 94.8%
# predictions_train = reg_utils.predict(train_X , train_Y , parameters)
# print('测试集: ') # 91.5%
# predictions_test = reg_utils.predict(test_X , test_Y , parameters)
#
# plt.title("Model without regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)
#
# # 可以看到曲线有点畸形,有过拟合的倾向
################################## 1.不使用正则化 ####################################


################################## 2.使用L2正则化 ####################################
# 要改的地方:1.cost的计算,2.反向传播的过程.
def compute_cost_with_regularization(A3 , Y , parameters , lambd):
    # L2正则化计算成本
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    # 分两个部分计算,加起来
    cross_entropy_cost = reg_utils.compute_cost(A3 , Y)

    L2_regularization_cost = 1/(2*m) * lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

    cost = cross_entropy_cost + L2_regularization_cost

    return cost

def backward_propagation_with_regularization(X, Y, cache, lambd):
    m = X.shape[1]

    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = (1 / m) * np.dot(dZ3, A2.T) + ((lambd * W3) / m)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1, X.T) + ((lambd * W1) / m)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

# 看看train起来的效果
# parameters = model(train_X , train_Y , lambd=0.5 , is_plot=True)
# print('使用正则化,训练集:') # 94.7%
# predictions_train = reg_utils.predict(train_X , train_Y , parameters)
# print('使用正则化,训练集:') # 94%
# predictions_test = reg_utils.predict(test_X , test_Y , parameters)
#
# plt.title("Model with L2-regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)
#
# # 可以看到曲线平滑了许多,泛化能力也上升了
################################## 2.使用L2正则化 ####################################


################################## 3.使用dropout ####################################
# 要改的地方:1.前向传播 2.反向传播

def forward_propagation_with_dropout(X , parameters , keep_prob = 0.5):
    # 让第一层和第二层随机删除一些节点,用np.random.rand()来初始化一个D
    # 如果D(l)低于keep_prob就置1,高于keep_prob就置0,然后element-wise一乘就行
    # 注意A(l)/keep_prob保持期望不变
    np.random.seed(1)

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']

    Z1 = np.dot(W1 , X) + b1
    A1 = reg_utils.relu(Z1)

    # 一层一层写吧,第一层
    D1 = np.random.rand(A1.shape[0] , A1.shape[1]) # np.random.rand:均匀分布
    D1 = D1 < keep_prob
    A1 = A1 * D1  # element-wise
    A1 = A1 / keep_prob

    Z2 = np.dot(W2 , A1) + b2
    A2 = reg_utils.relu(Z2)

    D2 = np.random.rand(A2.shape[0] , A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob

    Z3 = np.dot(W3 , A2) + b3
    A3 = reg_utils.sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3 , cache       # cache打包送到梯度回传


# 反向传播也只更新那些用到的节点,同样要给他们放大
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dA2 = dA2 * D2          # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA2 = dA2 / keep_prob   # 步骤2：缩放未舍弃的节点(不为0)的值

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)

    dA1 = dA1 * D1          # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA1 = dA1 / keep_prob   # 步骤2：缩放未舍弃的节点(不为0)的值

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients



# parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.03,is_plot=True,num_iterations=60000)
#
# print("使用随机删除节点，训练集:") # 93.8%
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print("使用随机删除节点，测试集:") # 95%
# reg_utils.predictions_test = reg_utils.predict(test_X, test_Y, parameters)
#
# plt.title("Model with dropout")
# axes = plt.gca()
# axes.set_xlim([-0.75, 0.40])
# axes.set_ylim([-0.75, 0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)
#

################################## 3.使用dropout ####################################


####################################### 第三部分：梯度校验 ######################################

################################## 1.一维线性 ####################################
def forward_propagation(x, theta):
    """

    实现图中呈现的线性前向传播（计算J）（J（theta）= theta * x）

    参数：
    x  - 一个实值输入
    theta  - 参数，也是一个实数

    返回：
    J  - 函数J的值，用公式J（theta）= theta * x计算
    """
    J = np.dot(theta, x)

    return J

# #测试forward_propagation
# print("-----------------测试forward_propagation-----------------")
# x, theta = 2, 4
# J = forward_propagation(x, theta)
# print ("J = " + str(J))

def backward_propagation(x, theta):  # theta的导数就是x(一维情况)
    """
    计算J相对于θ的导数。

    参数：
        x  - 一个实值输入
        theta  - 参数，也是一个实数

    返回：
        dtheta  - 相对于θ的成本梯度
    """
    dtheta = x

    return dtheta

# #测试backward_propagation
# print("-----------------测试backward_propagation-----------------")
# x, theta = 2, 4
# dtheta = backward_propagation(x, theta)
# print ("dtheta = " + str(dtheta))

def gradient_check(x, theta, epsilon=1e-7):
    """

    实现图中的反向传播。

    参数：
        x  - 一个实值输入
        theta  - 参数，也是一个实数
        epsilon  - 使用公式计算输入的微小偏移以计算近似梯度

    返回：
        近似梯度和后向传播梯度之间的差异
    """

    # 使用公式计算gradapprox。
    thetaplus = theta + epsilon                     # Step 1
    thetaminus = theta - epsilon                    # Step 2
    J_plus = forward_propagation(x, thetaplus)      # Step 3
    J_minus = forward_propagation(x, thetaminus)    # Step 4
    gradapprox = (J_plus - J_minus) / (2 * epsilon) # Step 5

    # 检查gradapprox是否足够接近backward_propagation()的输出
    grad = backward_propagation(x, theta)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference < 1e-7:
        print("梯度检查：梯度正常!")
    else:
        print("梯度检查：梯度超出阈值!")

    return difference

# #测试gradient_check
# print("-----------------测试gradient_check-----------------")
# x, theta = 2, 4
# difference = gradient_check(x, theta)
# print("difference = " + str(difference))

################################## 1.一维线性 ####################################

################################## 2.高维情况 ####################################
def forward_propagation_n(X, Y, parameters):
    """
    实现图中的前向传播（并计算成本）。

    参数：
        X - 训练集为m个例子
        Y -  m个示例的标签
        parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
            W1  - 权重矩阵，维度为（5,4）
            b1  - 偏向量，维度为（5,1）
            W2  - 权重矩阵，维度为（3,5）
            b2  - 偏向量，维度为（3,1）
            W3  - 权重矩阵，维度为（1,3）
            b3  - 偏向量，维度为（1,1）

    返回：
        cost - 成本函数（logistic）
    """
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = gc_utils.relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = gc_utils.relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = gc_utils.sigmoid(Z3)

    # 计算成本
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = (1 / m) * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache


def backward_propagation_n(X, Y, cache):
    """
    实现反向传播。

    参数：
        X - 输入数据点（输入节点数量，1）
        Y - 标签
        cache - 来自forward_propagation_n（）的cache输出

    返回：
        gradients - 一个字典，其中包含与每个参数、激活和激活前变量相关的成本梯度。
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1. / m) * np.dot(dZ3, A2.T)
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    # dW2 = 1. / m * np.dot(dZ2, A1.T) * 2  # Should not multiply by 2
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    # db1 = 4. / m * np.sum(dZ1, axis=1, keepdims=True) # Should not multiply by 4
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    """
    检查backward_propagation_n是否正确计算forward_propagation_n输出的成本梯度

    参数：
        parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
        grad_output_propagation_n的输出包含与参数相关的成本梯度。
        x  - 输入数据点，维度为（输入节点数量，1）
        y  - 标签
        epsilon  - 计算输入的微小偏移以计算近似梯度

    返回：
        difference - 近似梯度和后向传播梯度之间的差异
    """
    # 初始化参数
    parameters_values, keys = gc_utils.dictionary_to_vector(parameters)  # keys用不到,theta变成一竖条
    grad = gc_utils.gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # 计算gradapprox
    for i in range(num_parameters):
        # 计算J_plus [i]。输入：“parameters_values，epsilon”。输出=“J_plus [i]”
        thetaplus = np.copy(parameters_values)  # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon  # Step 2
        J_plus[i], cache = forward_propagation_n(X, Y, gc_utils.vector_to_dictionary(thetaplus))  # Step 3 ，cache用不到 , 从向量变回字典才能送进去前向传播

        # 计算J_minus [i]。输入：“parameters_values，epsilon”。输出=“J_minus [i]”。
        thetaminus = np.copy(parameters_values)  # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon  # Step 2
        J_minus[i], cache = forward_propagation_n(X, Y, gc_utils.vector_to_dictionary(thetaminus))  # Step 3 ，cache用不到

        # 计算gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    # 通过计算差异比较gradapprox和后向传播梯度。
    numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator  # Step 3'

    if difference < 1e-7:
        print("梯度检查：梯度正常!")
    else:
        print("梯度检查：梯度超出阈值!")

    return difference

# 测试高维导数检验
np.random.seed(2)
X = np.random.randn(4,3)
Y = np.array([1, 1,0])
W1 = np.random.randn(5,4)
b1 = np.random.randn(5,1)
W2 = np.random.randn(3,5)
b2 = np.random.randn(3,1)
W3 = np.random.randn(1,3)
b3 = np.random.randn(1,1)
parameters = {"W1":W1, "b1": b1,"W2": W2, "b2": b2,"W3": W3,"b3": b3}
cost, cache =  forward_propagation_n(X ,Y ,parameters)
gradients = backward_propagation_n(X,Y,cache)
difference =  gradient_check_n(parameters,gradients,X,Y,epsilon=1e-7)
print("difference = "  + str(difference))

################################## 2.高维情况 ####################################
####################################### 第三部分：梯度校验 ######################################