# 用最哈皮最稚嫩的语言搭建RNN和LSTM的前向/反向传播

import numpy as np
import rnn_utils  # 一些激活函数和优化器

# RNN单元
def rnn_cell_forward(xt , a_prev , parameters):
    '''
    定义一个RNN cell
    Args:
        xt: 时间 t的输入,(n_x , m)
        a_prev: 时间 t-1的隐藏状态,(n_a , m)
        parameters: 字典,包括参数: Wax , Waa , ba , Wya , by

    Returns:
        a_next: 时间 t的隐藏状态,(n_a , m)
        yt_pred: 时间 t的预测值,(n_y , m)
        cache: 反向传播需要的元组,包括(a_next , a_prev , xt , parameters)
    '''
    # 先从字典中取参数
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    ba = parameters['ba']
    Wya = parameters['Wya']
    by = parameters['by']
    # 计算输出值
    a_next = np.tanh(np.dot(Waa , a_prev) + np.dot(Wax , xt) + ba)
    yt_pred = rnn_utils.softmax(np.dot(Wya , a_next) + by)
    # 保存反向传播需要的值
    cache = (a_next , a_prev , xt , parameters)

    return a_next , yt_pred , cache


# # print('===========测试rnn_cell_forward============')
# np.random.seed(1)
# xt = np.random.randn(3 , 10) # n_x = 3 , m = 10
# a_prev = np.random.randn(5 , 10)
# Waa = np.random.randn(5 , 5)
# Wax = np.random.randn(5 , 3)
# Wya = np.random.randn(2 , 5)
# ba = np.random.randn(5 , 1)
# by = np.random.randn(2 , 1)
# # 设定变量的顺序很重要...不然和原程序结果不一样...
#
# parameters = {'Waa' : Waa , 'Wax' : Wax , 'ba' : ba , 'Wya' : Wya , 'by' : by}
#
#
# a_next , yt_pred , _ = rnn_cell_forward(xt , a_prev , parameters)
# print('a_next[4] = ' , a_next[4])
# print('a_next.shape = ' , a_next.shape)
# print('yt_pred[1] = ' , yt_pred[1])
# print('yt_pred.shape = ' , yt_pred.shape)


# RNN前向传播部分
def rnn_forward(x , a0 , parameters):
    '''
    用上面定义的单步传播循环,组装成前向传播到底的函数.
    Args:
        x: 所有输入数据,(n_x , m , T_x)
        a0: 初始化的隐藏状态,(n_a , m)
        parameters: 字典,包括: Wax,Waa,ba,Wya,by

    Returns:
        a: 所有时间步的隐藏状态,(n_a , m , T_x)
        y_pred: 所有时间步的预测值,(n_y , m , T_y) , 本例中 T_y = T_x
        caches(列表): 形式如: (cache , x)

    '''
    # 初始化caches
    caches = []
    # 通过x和Wya获取维度信息
    n_x , m , T_x = x.shape
    n_y , n_a = parameters['Wya'].shape
    # 初始化a和y_pred(定个型)
    a = np.zeros((n_a , m , T_x))
    y_pred = np.zeros((n_y , m , T_x))

    # 初始化a_next
    a_next = a0

    # 从左到右遍历时间步
    for t in range(T_x):
        # 1.用单步rnn更新a_next,输出yt_pred
        a_next , yt_pred , cache = rnn_cell_forward(x[ :, : , t] , a_next , parameters)

        # 2.把a_next存进a
        a[ : ,:, t] = a_next

        # 3.把yt_pred存进y_pred
        y_pred[ :, :, t] = yt_pred

        # 4.把cache存进caches
        caches.append(cache)

    # 5.保留所有反向传播需要的参数
    caches = (caches , x)

    return a , y_pred , caches

# print('==========测试rnn_forward==========')
# np.random.seed(1)
# x = np.random.randn(3,10,4)
# a0 = np.random.randn(5,10)
# Waa = np.random.randn(5,5)
# Wax = np.random.randn(5,3)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
#
# a, y_pred, caches = rnn_forward(x, a0, parameters)
# print("a[4][1] = ", a[4][1])
# print("a.shape = ", a.shape)
# print("y_pred[1][3] =", y_pred[1][3])
# print("y_pred.shape = ", y_pred.shape)
# print("caches[1][1][3] =", caches[1][1][3])
# print('x[1][3] = ' , x[1][3]) # 果然和上面一样
# print("len(caches) = ", len(caches))

##################### RNN前向传播函数搭建完毕 ##########################

# LSTM部分
def lstm_cell_forward(xt , a_prev , c_prev , parameters):
    '''
    老套路,先写一个cell的前向传递,在用循环实现整体前向传播
    Args:
        xt: 时间步t的输入,(n_x , m)
        a_prev: 上一时间步t-1的隐藏状态,(n_a , m)
        c_prev: 上一时间步t-1的记忆状态,(n_a , m)
        parameters: 字典,包括5组 W和 b:
                        Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                        bf -- 遗忘门的偏置，维度为(n_a, 1)
                        Wi -- 更新门的权值，维度为(n_a, n_a + n_x)
                        bi -- 更新门的偏置，维度为(n_a, 1)
                        Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                        bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                        Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                        bo -- 输出门的偏置，维度为(n_a, 1)
                        Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                        by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)

    Returns:
        a_next: 时间t的隐藏状态,(n_a , m)
        c_next: 时间t的记忆状态,(n_a , m)
        yt_pred: 时间t的预测值,(n_y , m)
        cache: 反向传播需要的参数,包括(a_next , c_next , a_prev , c_prev , xt , parameters)

    '''
    # 先从字典中取出相关值
    Wf = parameters['Wf']
    bf = parameters['bf']
    Wi = parameters['Wi']
    bi = parameters['bi']
    Wc = parameters['Wc']
    bc = parameters['bc']
    Wo = parameters['Wo']
    bo = parameters['bo']
    Wy = parameters['Wy']
    by = parameters['by']

    # 获取形状信息
    n_x , m = xt.shape
    n_y , n_a = Wy.shape
    # 把a_prev和xt竖着粘起来
    contact = np.zeros((n_a + n_x , m))
    contact[ : n_a, : ] = a_prev
    contact[n_a : , : ] = xt

    # 按公式计算各值
    # 遗忘门: ft
    ft = rnn_utils.sigmoid(np.dot(Wf , contact) + bf)
    # 更新门: it
    it = rnn_utils.sigmoid(np.dot(Wi , contact) + bi)
    # 待更新的备选cct
    cct = np.tanh(np.dot(Wc , contact) + bc)
    # c_next
    c_next = ft * c_prev + it * cct
    # 输出门: ot
    ot = rnn_utils.sigmoid(np.dot(Wo , contact) + bo)
    # a_next
    a_next = ot * np.tanh(c_next)
    # 预测值: yt_pred
    yt_pred = rnn_utils.softmax(np.dot(Wy , a_next) + by)
    # cache打包好反向传播的参数
    cache = (a_next , c_next , a_prev , c_prev , ft , it , cct , ot , xt , parameters)
    return a_next , c_next , yt_pred , cache

# print('============测试lstm_cell_forward=============')
# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# c_prev = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
# print("a_next[4] = ", a_next[4])
# print("a_next.shape = ", a_next.shape)
# print("c_next[3] = ", c_next[3])
# print("c_next.shape = ", c_next.shape)
# print("yt[1] =", yt[1])
# print("yt.shape = ", yt.shape)
# print("cache[1][3] =", cache[1][3])  # 应该和c_next[3]相等,确实
# print("len(cache) = ", len(cache))


# LSTM前向传播部分
def lstm_forward(x , a0 , parameters): # a0可以不是0向量,但是代表记忆的c0肯定是0向量,所以不需要输入
    '''
    用for循环把lstm单元拼起来
    Args:
        x: 所有时间步的输入数据,(n_x , m , T_x)
        a0: 初始化隐藏状态,(n_a , m)
        parameters: 字典,包括各种 W 和 b

    Returns:
        a: 所有时间步的隐藏状态,(n_a , m , T_x)
        y: 所有时间步的预测值,(n_y , m , T_y),本例中 T_y = T_x
        c: 所有时间步的记忆状态,(n_a , m , T_x)
        caches: [cache , x]
    '''
    # 初始化caches
    caches = []

    # 维度获取(思维窃取)
    n_x , m , T_x = x.shape
    n_y , n_a = parameters['Wy'].shape

    # 给a、c、y定型
    a = np.zeros((n_a , m , T_x))
    c = np.zeros((n_a , m , T_x))
    y = np.zeros((n_y , m , T_x))

    # 初始化a_next , c_next
    a_next = a0
    c_next = np.zeros((n_a , m))

    # 遍历所有时间步
    for t in range(T_x):
        # 走一步存四下
        a_next , c_next , yt_pred , cache = lstm_cell_forward(x[ : , : ,t] , a_next , c_next , parameters)
        a[ : , : ,t] = a_next
        y[ : , : ,t] = yt_pred
        c[ : , : ,t] = c_next
        caches.append(cache)

    caches = (caches , x)
    return a , y , c , caches

# print('=============测试lstm_forward=============')
# np.random.seed(1)
# x = np.random.randn(3,10,7)
# a0 = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a, y, c, caches = lstm_forward(x, a0, parameters)
# print("a[4][3][6] = ", a[4][3][6])
# print("a.shape = ", a.shape)
# print("y[1][4][3] =", y[1][4][3])
# print("y.shape = ", y.shape)
# print("caches[1][1][1] =", caches[1][1][1])
# print('x[1][1] = ' , x[1][1])
# print("c[1][2][1]", c[1][2][1])
# print("len(caches) = ", len(caches))

##################### LSTM前向传播函数搭建完毕 ##########################

# RNN反向传播部分,直接复制了,太nm复杂了
# 单步反向传播
def rnn_cell_backward(da_next, cache):
    """
    实现基本的RNN单元的单步反向传播

    参数：
        da_next -- 关于下一个隐藏状态的损失的梯度。
        cache -- 字典类型，rnn_step_forward()的输出

    返回：
        gradients -- 字典，包含了以下参数：
                        dx -- 输入数据的梯度，维度为(n_x, m)
                        da_prev -- 上一隐藏层的隐藏状态，维度为(n_a, m)
                        dWax -- 输入到隐藏状态的权重的梯度，维度为(n_a, n_x)
                        dWaa -- 隐藏状态到隐藏状态的权重的梯度，维度为(n_a, n_a)
                        dba -- 偏置向量的梯度，维度为(n_a, 1)
    """
    # 获取cache 的值
    a_next, a_prev, xt, parameters = cache

    # 从 parameters 中获取参数
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # 计算tanh相对于a_next的梯度.
    dtanh = (1 - np.square(a_next)) * da_next

    # 计算关于Wax损失的梯度
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    # 计算关于Waa损失的梯度
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    # 计算关于b损失的梯度
    dba = np.sum(dtanh, keepdims=True, axis=-1)

    # 保存这些梯度到字典内
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients


# 连续RNN反向传播
def rnn_backward(da, caches):
    """
    在整个输入数据序列上实现RNN的反向传播

    参数：
        da -- 所有隐藏状态的梯度，维度为(n_a, m, T_x)
        caches -- 包含向前传播的信息的元组

    返回：
        gradients -- 包含了梯度的字典：
                        dx -- 关于输入数据的梯度，维度为(n_x, m, T_x)
                        da0 -- 关于初始化隐藏状态的梯度，维度为(n_a, m)
                        dWax -- 关于输入权重的梯度，维度为(n_a, n_x)
                        dWaa -- 关于隐藏状态的权值的梯度，维度为(n_a, n_a)
                        dba -- 关于偏置的梯度，维度为(n_a, 1)
    """
    # 从caches中获取第一个cache（t=1）的值
    caches, x = caches
    a1, a0, x1, parameters = caches[0]

    # 获取da与x1的维度信息
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # 初始化梯度
    dx = np.zeros([n_x, m, T_x])
    dWax = np.zeros([n_a, n_x])
    dWaa = np.zeros([n_a, n_a])
    dba = np.zeros([n_a, 1])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])

    # 处理所有时间步
    for t in reversed(range(T_x)):
        # 计算时间步“t”时的梯度
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])

        # 从梯度中获取导数
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients[
            "dWaa"], gradients["dba"]

        # 通过在时间步t添加它们的导数来增加关于全局导数的参数
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    # 将 da0设置为a的梯度，该梯度已通过所有时间步骤进行反向传播
    da0 = da_prevt

    # 保存这些梯度到字典内
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients

##################### RNN反向传播函数搭建完毕 ##########################

# LSTM单步反向传播
def lstm_cell_backward(da_next, dc_next, cache):
    """
    实现LSTM的单步反向传播

    参数：
        da_next -- 下一个隐藏状态的梯度，维度为(n_a, m)
        dc_next -- 下一个单元状态的梯度，维度为(n_a, m)
        cache -- 来自前向传播的一些参数

    返回：
        gradients -- 包含了梯度信息的字典：
                        dxt -- 输入数据的梯度，维度为(n_x, m)
                        da_prev -- 先前的隐藏状态的梯度，维度为(n_a, m)
                        dc_prev -- 前的记忆状态的梯度，维度为(n_a, m, T_x)
                        dWf -- 遗忘门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbf -- 遗忘门的偏置的梯度，维度为(n_a, 1)
                        dWi -- 更新门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbi -- 更新门的偏置的梯度，维度为(n_a, 1)
                        dWc -- 第一个“tanh”的权值的梯度，维度为(n_a, n_a + n_x)
                        dbc -- 第一个“tanh”的偏置的梯度，维度为(n_a, n_a + n_x)
                        dWo -- 输出门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbo -- 输出门的偏置的梯度，维度为(n_a, 1)
    """
    # 从cache中获取信息
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    # 获取xt与a_next的维度信息
    n_x, m = xt.shape
    n_a, m = a_next.shape

    # 根据公式7-10来计算门的导数
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

    # 根据公式11-14计算参数的导数
    concat = np.concatenate((a_prev, xt), axis=0).T
    dWf = np.dot(dft, concat)
    dWi = np.dot(dit, concat)
    dWc = np.dot(dcct, concat)
    dWo = np.dot(dot, concat)
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)

    # 使用公式15-17计算洗起来了隐藏状态、先前记忆状态、输入的导数。
    da_prev = np.dot(parameters["Wf"][:, :n_a].T, dft) + np.dot(parameters["Wc"][:, :n_a].T, dcct) + np.dot(
        parameters["Wi"][:, :n_a].T, dit) + np.dot(parameters["Wo"][:, :n_a].T, dot)

    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next

    dxt = np.dot(parameters["Wf"][:, n_a:].T, dft) + np.dot(parameters["Wc"][:, n_a:].T, dcct) + np.dot(
        parameters["Wi"][:, n_a:].T, dit) + np.dot(parameters["Wo"][:, n_a:].T, dot)

    # 保存梯度信息到字典
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients

# LSTM连续反向传播
def lstm_backward(da, caches):
    """
    实现LSTM网络的反向传播

    参数：
        da -- 关于隐藏状态的梯度，维度为(n_a, m, T_x)
        cachses -- 前向传播保存的信息

    返回：
        gradients -- 包含了梯度信息的字典：
                        dx -- 输入数据的梯度，维度为(n_x, m，T_x)
                        da0 -- 先前的隐藏状态的梯度，维度为(n_a, m)
                        dWf -- 遗忘门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbf -- 遗忘门的偏置的梯度，维度为(n_a, 1)
                        dWi -- 更新门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbi -- 更新门的偏置的梯度，维度为(n_a, 1)
                        dWc -- 第一个“tanh”的权值的梯度，维度为(n_a, n_a + n_x)
                        dbc -- 第一个“tanh”的偏置的梯度，维度为(n_a, n_a + n_x)
                        dWo -- 输出门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbo -- 输出门的偏置的梯度，维度为(n_a, 1)

    """

    # 从caches中获取第一个cache（t=1）的值
    caches, x = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    # 获取da与x1的维度信息
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # 初始化梯度
    dx = np.zeros([n_x, m, T_x])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])
    dc_prevt = np.zeros([n_a, m])
    dWf = np.zeros([n_a, n_a + n_x])
    dWi = np.zeros([n_a, n_a + n_x])
    dWc = np.zeros([n_a, n_a + n_x])
    dWo = np.zeros([n_a, n_a + n_x])
    dbf = np.zeros([n_a, 1])
    dbi = np.zeros([n_a, 1])
    dbc = np.zeros([n_a, 1])
    dbo = np.zeros([n_a, 1])

    # 处理所有时间步
    for t in reversed(range(T_x)):
        # 使用lstm_cell_backward函数计算所有梯度
        gradients = lstm_cell_backward(da[:, :, t], dc_prevt, caches[t])
        # 保存相关参数
        dx[:, :, t] = gradients['dxt']
        dWf = dWf + gradients['dWf']
        dWi = dWi + gradients['dWi']
        dWc = dWc + gradients['dWc']
        dWo = dWo + gradients['dWo']
        dbf = dbf + gradients['dbf']
        dbi = dbi + gradients['dbi']
        dbc = dbc + gradients['dbc']
        dbo = dbo + gradients['dbo']
    # 将第一个激活的梯度设置为反向传播的梯度da_prev。
    da0 = gradients['da_prev']

    # 保存所有梯度到字典变量内
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients

##################### LSTM反向传播函数搭建完毕 ##########################
