'''
本程序中包括两个例子 : 一个两层神经网络,一个多层神经网络,比较它们分类猫图片的准确率
我们来说一下步骤：
    1. 初始化网络参数
    2. 前向传播
        2.1 计算一层的中线性求和的部分
        2.2 计算激活函数的部分（ReLU使用L-1次，Sigmod使用1次）
        2.3 结合线性求和与激活函数
    3. 计算误差
    4. 反向传播
        4.1 线性部分的反向传播公式
        4.2 激活函数部分的反向传播公式
        4.3 结合线性部分与激活函数的反向传播公式
    5. 更新参数
'''
import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases # 用于检验各个模块是否工作
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils # 数据导入

np.random.seed(1)

#######################################初始化部分############################################
# 初始化函数(两层)
def initialize_parameters(n_x , n_h , n_y):
    '''
    初始化两层网络参数: W , b
    :param n_x: 输入层节点数
    :param n_h: 隐藏层节点数
    :param n_y: 输出层节点数
    :return: 字典parameters , 包括 W1 , b1 , W2 , b2
    '''
    W1 = np.random.randn(n_h , n_x) * 0.01
    b1 = np.zeros((n_h , 1))
    W2 = np.random.randn(n_y , n_h) * 0.01
    b2 = np.zeros((n_y , 1))

    # 用断言保证格式正确
    assert(W1.shape == (n_h , n_x))
    assert(b1.shape == (n_h , 1))
    assert(W2.shape == (n_y , n_h))
    assert(b2.shape == (n_y , 1))

    parameters = {
        'W1' : W1,
        'b1' : b1,
        'W2' : W2,
        'b2' : b2,
    }

    return parameters

# print('===========测试initialize_parameters==============')
# parameters_ = initialize_parameters(3 , 2 , 1)
# print('W1 = ' , parameters_['W1'])
# print('b1 = ' , parameters_['b1'])
# print('W2 = ' , parameters_['W2'])
# print('b2 = ' , parameters_['b2'])


# 初始化函数(deep),使用for循环来随机初始值
def initialize_parameters_deep(layers_dims):
    '''
    初始化多层神经网络参数的函数
    :param layers_dims: 包含网络中每层的节点数量的列表
    :return: parameters : 包含'W1','b1',...'WL','bL'的字典
    '''
    np.random.seed(3)
    parameters = {}  # 好好学怎么用循环在字典里加 KEY
    L = len(layers_dims)

    for l in range(1 , L):
        # 这里两种写法先观望一下
        parameters['W' + str(l)] = np.random.randn(layers_dims[l] , layers_dims[l-1]) / np.sqrt(layers_dims[l-1])
        # 这一种是实际训练用的,用下面那种loss会卡在0.64
        #parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        # 这一种是testcases用的
        parameters['b' + str(l)] = np.zeros((layers_dims[l] , 1))  # zeros要用两个括号嗷!!!!!!!!!!!!!!!!!

        # 用断言保证格式
        assert(parameters['W' + str(l)].shape == (layers_dims[l] , layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layers_dims[l] , 1))

    return parameters


# print('============测试initialize_parameters_deep================')
# parameters_ = initialize_parameters_deep([5,4,3])
# print(parameters_)

#######################################初始化部分############################################

#######################################前向传播部分###########################################
# 1.线性部分(linear)
def linear_forward(A , W , b):
    '''
    实现前向传播的线性部分
    :param A: 上一层激活层的输出 A(l-1)
    :param W: 这一层的 W(l)
    :param b: 这一层的 b(l)
    :return: Z:这一层激活函数的输入 , cache:一个包含 A, W, b 的元组(存下来用来反向更新的)
    '''
    Z = np.dot(W , A) + b
    assert(Z.shape == (W.shape[0] , A.shape[1]))  # Z(l) shape:(n(l) , m)
    cache = (A , W , b)

    return Z , cache

# print('================测试linear_forward===============')
# A_ , W_ , b_ = testCases.linear_forward_test_case()
# Z_ , _ = linear_forward(A_ , W_ , b_)
# print('Z = ' , Z_)

# 2.线性激活部分(linear -> activation)
def linear_activation_forward(A_prev , W , b , activation):
    '''
    实现linear -> activation 这一层的前向传播
    :param A_prev: 上一层激活层的输出
    :param W: W(l)
    :param b: b(l)
    :param activation: 选择在这一层的激活层名,字符串类型('sigmoid' or 'relu')
    :return: A(l):激活层的输出 , cache:(linear_cache(A , W , b) , activation_cache(Z)) 用于计算反向传递
    '''
    Z , linear_cache = linear_forward(A_prev , W , b)
    if activation == 'sigmoid':
        A , activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A , activation_cache = relu(Z)
    assert(A.shape == (W.shape[0] , A_prev.shape[1]))
    cache = (linear_cache , activation_cache)
    return A , cache

# print('============测试linear_activation_forward=============')
# A_prev_ , W_ , b_ = testCases.linear_activation_forward_test_case()
#
# A_ , _ = linear_activation_forward(A_prev_ , W_ , b_ , activation='sigmoid')
# print('sigmoid , A = ' , A_)
#
# A_ , _ = linear_activation_forward(A_prev_ , W_ , b_ , activation='relu')
# print('ReLU , A = ' , A_)

# 3.多层前向传播部分
def L_model_forward(X , parameters):
    '''
    实现 (linear -> relu)*(L-1) -> linear -> sigmoid 计算前向传播
    :param X: A(l-1) shape: (n(l-1) , m)
    :param parameters: 包含 'Wl','bl'的字典,最开始由 initialize_parameters_deep()得来
    :return: AL : 最后输出值(Yhat)用于求 L(Y , Yhat) ,
             caches :列表,linear_activation_forward的输出cache, L-1个relu的 , 1个sigmoid的
    '''
    caches = []
    A = X
    L = len(parameters) // 2                # 有 W 有 b
    for l in range(1 , L):                  # L层网络激活L-1次
        A_prev = A
        A , cache = linear_activation_forward(A_prev , parameters['W'+str(l)] , parameters['b' + str(l)] , 'relu')
        caches.append(cache)
    # 每个 cache都有 4个 array
    AL , cache = linear_activation_forward(A , parameters['W' + str(L)] , parameters['b' + str(L)] , 'sigmoid')
    caches.append(cache)

    assert(AL.shape == (1 , X.shape[1]))    # (1 , m)

    return AL , caches

# print('==============测试L_model_forward===============')
# X_ , parameters_ = testCases.L_model_forward_test_case()
# AL_ , caches_ = L_model_forward(X_ , parameters_)
# print('输出AL = '  , AL_)
# print('caches = ' , caches_)
# print('神经网络层数 : ' , len(caches_))

# 4.计算成本
def compute_cost(AL , Y):
    '''
    分类采用交叉熵成本
    :param AL: 预测值
    :param Y: 标签值
    :return: cost - 交叉熵成本
    '''
    m = Y.shape[1]
    # multiply : 对位乘 , cost越小越好
    cost = (-1/m) * np.sum(np.multiply(Y , np.log(AL)) + np.multiply((1 - Y) , np.log(1 - AL)))
    cost = np.squeeze(cost) # 变成一个数字
    assert(cost.shape == ())
    return cost

# print('===========测试compute_cost==============')
# Y_ , AL_ = testCases.compute_cost_test_case()
# print('cost = ' , compute_cost(AL_ , Y_))

#######################################前向传播部分###########################################

#######################################反向传播部分###########################################
# 1.反向传播线性部分
def linear_backward(dZ , cache):
    '''
    为单层实现反向传播的线性部分,输入dZ(l)输出dA(l-1),dW(l),db(l)
    :param dZ: dZ(l)
    :param cache: 来自当前层线性前向传播的元组 (A_prev , W , b)
    :return: dA(l-1) , dW(l) , db(l)
    '''
    A_prev , W , b = cache
    m = A_prev.shape[1]
    dW = (1/m) * np.dot(dZ , A_prev.T)
    db = (1/m) * np.sum(dZ , axis=1 , keepdims=True)
    dA_prev = np.dot(W.T , dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev , dW , db

# print('===============测试linear_backward==============')
# dZ_ , cache_ = testCases.linear_backward_test_case()
# dA_prev_ , dW_ , db_ = linear_backward(dZ_ , cache_)
# print('dA_prev = ' , dA_prev_)
# print('dW = ' , dW_)
# print('db = ' , db_)

# 2.反向线性激活部分
def linear_activation_backward(dA , cache , activation = 'relu'):
    '''
    实现激活层的求导, 输入dA(l)输出dZ(l),再通过dZ(l)求dA,dW,db
    :param dA: 这一层的激活层的输出A(l)
    :param cache: 对应部分的cache, cache:(linear_cache(A , W , b) , activation_cache(Z))
    :param activation: 激活函数选择
    :return: dA_prev , dW , db
    '''
    linear_cache , activation_cache = cache
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA , activation_cache) # 这里只用activation_cache(Z)就能求出 dZ了
        dA_prev , dW , db = linear_backward(dZ , linear_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA , activation_cache)
        dA_prev , dW , db = linear_backward(dZ , linear_cache)

    return dA_prev , dW , db

# print('==================测试linear_activation_backward===================')
# dA_ , cache_ = testCases.linear_activation_backward_test_case()
# dA_prev_ , dW_ , db_ = linear_activation_backward(dA_ , cache_ , activation='sigmoid')
# print('sigmoid : ')
# print('dA_prev = ' , dA_prev_)
# print('dW = ' , dW_)
# print('db = ' , db_)
# dA_prev_ , dW_ , db_ = linear_activation_backward(dA_ , cache_ , activation='relu')
# print('relu : ')
# print('dA_prev = ' , dA_prev_)
# print('dW = ' , dW_)
# print('db = ' , db_)

# 3.多层反向传播部分
def L_model_backward(AL , Y , caches):
    '''
    反向传播 : [linear <- relu]*(L-1) <- linear <- sigmoid
    :param AL: 预测值
    :param Y: 标签值(AL , Y)用来算 dAL
    :param caches: 对应部分的输出,列表,linear_activation_forward的输出cache, L-1个relu的 , 1个sigmoid的 , 长度为层数
    :return: 字典grads,包括grads['dAl'],grads['dWl'],grads['dbl']
    '''
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - np.divide(Y , AL) + np.divide((1 - Y) , (1 - AL))

    current_cache = caches[L-1]
    grads['dA'+str(L)],grads['dW'+str(L)],grads['db'+str(L)] = linear_activation_backward(dAL , current_cache , 'sigmoid')

    # 启动循环relu层
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA' + str(l+1)] , grads['dW' + str(l+1)] , grads['db' + str(l+1)] = \
        linear_activation_backward(grads['dA' + str(l+2)] , current_cache , 'relu')

    return grads

# print('=============测试L_model_backward==============')
# AL_ , Y_ , caches_ = testCases.L_model_backward_test_case()
# grads_ = L_model_backward(AL_ , Y_ , caches_)
# print(grads_)

# 4.更新参数
def update_parameters(parameters , grads , learning_rate):
    '''
    使用梯度下降更新参数
    :param parameters: 字典,待更新的参数
    :param grads: 字典,包含各参数梯度
    :param learning_rate: 学习率
    :return: parameters : 更新好的参数
    '''
    L = len(parameters) // 2
    for l in range(L):
        parameters['W' + str(l+1)] -= learning_rate * grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] -= learning_rate * grads['db' + str(l+1)]
    return parameters

# print('============测试update_parameters============')
# parameters_ , grads_ = testCases.update_parameters_test_case()
# parameters = update_parameters(parameters_ , grads_ , 0.1)
# print(parameters)

#######################################反向传播部分###########################################

#######################################两层神经网络###########################################
def two_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False,isPlot=True):
    '''
    实现一个两层的神经网络,只需要用到前向和后向的二技能就够了,三技能是给 DNN用的
    :param X: 输入数据
    :param Y: 标签
    :param layers_dims: 每层的神经元数:(n_x , n_h , n_y)
    :param learning_rate: 学习率
    :param num_iterations: 迭代次数
    :param print_cost: 每100次迭代打印一次cost
    :param isPlot: 判断是否绘图(cost)
    :return: parameters,包含W1,b1,W2,b2的字典
    '''
    np.random.seed(1)
    grads = {}
    costs = []
    (n_x , n_h , n_y) = layers_dims
    # 初始化参数
    parameters = initialize_parameters(n_x , n_h , n_y)

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # 开始训练
    for i in range(num_iterations):
        # 前向传播,这里的cache是包括4个数的((A , W , b) , Z)
        A1 , cache1 = linear_activation_forward(X , W1 , b1 , activation='relu')
        A2 , cache2 = linear_activation_forward(A1 , W2 , b2 , activation='sigmoid')

        # 计算成本cost
        cost = compute_cost(A2 , Y)

        # 反向传播
        # 先初始化输出位置的导数
        dA2 = -np.divide(Y , A2) + np.divide((1 - Y) , (1 - A2))

        # 反向传播两层,输出dA,dW,db
        dA1 , dW2 , db2 = linear_activation_backward(dA2 , cache2 , activation='sigmoid')
        _ , dW1 , db1 = linear_activation_backward(dA1 , cache1 , activation='relu') # dA0没必要求

        # 用字典grads把导数包起来
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # 更新参数
        parameters = update_parameters(parameters , grads , learning_rate)

        # 如果 print_cost = True的话打印成本
        if i % 100 == 0:
            # 记录成本
            costs.append(cost)
            # 判断是否打印成本值
            if print_cost:
                print('第' , i , '次迭代的成本为: ' , cost)

    # 训练完了画图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations(per 100)')
        plt.title('Learning rate = ' + str(learning_rate))
        plt.show()

    return parameters

# 载入数据
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0] , -1).T  # shape : (n_x , m)
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0] , -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

# 开始训练
# layers_dims = (64*64*3 , 7 , 1)
# parameters = two_layer_model(train_x , train_y , layers_dims , num_iterations=2500 , print_cost=True , isPlot=True)
# 自己写的训练起来是真滴慢吖...

# 构建预测函数
def predict(X , y , parameters):
    '''
    用于预测 L层神经网络的结果,当然也包括 2层
    :param X: 测试集
    :param y: 标签
    :param parameters: W(l),b(l)
    :return: p - 给定数据集 X的预测
    '''
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1 , m))               # zeros一定要两个括号不然报错 : TypeError: data type not understood

    # 根据参数前向传播
    probas , _ = L_model_forward(X , parameters)

    for i in range(probas.shape[1]):    # 这里不能用 round,爷试过了
        if probas[0 , i] > 0.5:
            p[0 , i] = 1
        else:
            p[0 , i] = 0

    print('准确率为 : ' + str(float(np.sum((p == y)) / m)))
    return p

# print('两层网络训练集 : ')
# _ = predict(train_x , train_y , parameters)
# print('两层网络测试集 : ')
# _ = predict(test_x , test_y , parameters)
# 测试集上表现:准确率72% , 比起一层的逻辑回归(70%)有进步
#######################################两层神经网络###########################################

#######################################多层神经网络###########################################
def L_layer_model(X , Y , layers_dims , learning_rate = 0.0075 , num_iterations = 3000 , print_cost = False , isPlot = True):
    '''
    训练一个L层的神经网络
    :param X: 输入数据
    :param Y: 标签
    :param layers_dims: 每层神经元数量
    :param learning_rate: 学习率
    :param num_iteration: 迭代次数
    :param print_cost: 每100次迭代打印一次cost
    :param isPlot: 是否画cost图
    :return: parameters - 训练好的参数(可以用来预测)
    '''
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        # 前向传播
        AL , caches = L_model_forward(X , parameters) # 这里的caches是个列表,里面装的每个元素都是长度为 4
        # 计算cost
        cost = compute_cost(AL , Y)
        # 计算梯度反向传播
        grads = L_model_backward(AL , Y , caches)
        parameters = update_parameters(parameters , grads , learning_rate)

        # 如果 print_cost = True的话打印成本
        if i % 100 == 0:
            # 记录成本
            costs.append(cost)
            # 判断是否打印成本值
            if print_cost:
                print('第' , i , '次迭代的成本为: ' , cost)

    # 训练完了画图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations(per 100)')
        plt.title('Learning rate = ' + str(learning_rate))
        plt.show()

    return parameters

# 载入数据
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0] , -1).T  # shape : (n_x , m)
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0] , -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

# 开始训练
layers_dims = (64*64*3 , 20 , 7 , 5 , 1) # 4层网络(输入不算)
parameters = L_layer_model(train_x , train_y , layers_dims , num_iterations=2500 , print_cost=True , isPlot=True)

print('四层网络训练集 : ')
_ = predict(train_x , train_y , parameters)
print('四层网络测试集 : ')
pred_test = predict(test_x , test_y , parameters)
# 测试集准确率:78%,又有进步了

#######################################多层神经网络###########################################

# 观察在 L层神经网络中被分错的图片
def print_mislabeled_images(classes , X , y , p):
    '''
    绘制猜测错误的图片
    :param classes: 数据自带的标签,cat or noncat
    :param X: 数据集
    :param y: 标签
    :param p: 预测值
    :return: 无,打印预测错误的图片
    '''
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0 , 40.0)  # 图片的长宽
    num_images = len(mislabeled_indices[0])  # 预测错的图片数量
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
    plt.show()

print_mislabeled_images(classes , test_x , test_y , pred_test)


# 用自己的图片试试模型
import imageio # 用来看图的
import scipy.misc
'''
官方scipy中提到，imread is deprecated! imread is deprecated in SciPy 1.0.0,
 and will be removed in 1.2.0. Use imageio.imread instead.
SciPy1.0.0不赞成使用imread，在1.2中已经弃用，可以使用imageio.imread来代替
'''
#my_image = '灰猫.jpg'
my_image = '哆啦A梦.jpg'
my_label_y = [1]
fname = my_image
image = np.array(imageio.imread(fname))
num_px = 64
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
print('本地图片预测')
my_predicted_image = predict(my_image, my_label_y, parameters)
plt.imshow(image)
plt.show()

print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")