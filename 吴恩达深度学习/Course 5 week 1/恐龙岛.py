'''
    欢迎来到恐龙岛，恐龙生活于在6500万年前，现在研究人员在试着复活恐龙，而你的任务就是给恐龙命名，
    如果一只恐龙不喜欢它的名字，它可能会狂躁不安，所以你要谨慎选择。
    用cllm_utils里现成的RNN函数搭建一个RNN网络来给恐龙起名字
'''
import numpy as np
import random
import time
import cllm_utils

############################# 数据预处理 ################################
# 先读取数据,已有的恐龙名字
data = open('dinos.txt' , 'r').read()

# 全换成小写字母
data = data.lower()
# 记录出现过的字母
chars = list(set(data))
# 获取大小信息
data_size , vocab_size = len(data) , len(chars)

print(chars) # 有个'\n'在里面,起<EOS>的作用
print('共有%d个名字,由%d个字符组成' % (data_size , vocab_size))

# 搞两个字典,实现字符与索引的互换
char_to_ix = {ch : i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i : ch for i, ch in enumerate(sorted(chars))}

print(char_to_ix)
print(ix_to_char)

############################# 数据预处理 ################################

###################### 模块构建,包括梯度修剪&采样模块 ##########################
# 1.梯度修剪模块
def clip(gradients , maxValue):
    '''
    用np.clip来把梯度框在一个范围里,防止梯度爆炸,大于maxValue就直接等于maxValue.
    Args:
        gradients: 各种 dW , db
        maxValue: 阈值

    Returns:
        gradients: 修剪后的梯度

    '''
    # 先把参数拿出来
    dWaa , dWax , dWya , db , dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    # 修剪
    for gradient in [dWaa , dWax , dWya , db , dby]:
        np.clip(gradient , -maxValue , maxValue , out=gradient)

    gradients = {'dWaa':dWaa , 'dWax':dWax , 'dWya':dWya , 'db':db , 'dby':dby}

    return gradients

# 2.采样模块,一个一个生成: x<t+1> = y<t>
def sample(parameters , char_to_ix , seed):
    '''
    在一个已经训练好的模型上实现生成连续序列
    Args:
        parameters: 包含 Waa,Wax,Wya,b,by的字典
        char_to_ix: 字符到索引的字典
        seed: 随机种子

    Returns:
        indices: 包含采样字符的索引的列表
    '''
    # 取参数
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    # 获取长度
    vocab_size = by.shape[0] # n_y
    n_a = Waa.shape[1]

    # 1.初始化定型
    x = np.zeros((vocab_size , 1))  # one-hot,x<0>令为0向量,这样才能随机产生名字
    a_prev = np.zeros((n_a , 1))  # a<0>定为0向量
    indices = []
    idx = -1  # 用于检测换行符

    counter = 0  # 到50个字符就强制结束,防止无限循环
    newline_character = char_to_ix['\n']

    # 循环抽签,抽一个往indices里放一个
    while (idx != newline_character and counter < 50):
        # 2.前向传播
        a = np.tanh(np.dot(Waa , a_prev) + np.dot(Wax , x) + b)
        y = cllm_utils.softmax(np.dot(Wya , a) + by)

        # 设定随机种子
        np.random.seed(counter + seed)
        # 3.按概率分布抽取字符的索引
        idx = np.random.choice(list(range(vocab_size)) , p=y.ravel())
        # 抽一个往列表里加一个
        indices.append(idx)
        # x<t+1> = y<t>
        x = np.zeros((vocab_size , 1))
        x[idx] = 1 # one-hot
        # 更新a_prev
        a_prev = a
        # 累加器
        seed += 1
        counter += 1

    if counter == 50:  # 加个尾巴
        indices.append(newline_character)

    return indices

# print('=============测试sample===============')
# np.random.seed(2)
# _, n_a = 20, 100
# Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
#
#
# indices = sample(parameters, char_to_ix, 0)
# print("Sampling:")
# print("list of sampled indices:", indices)
# print("list of sampled characters:", [ix_to_char[i] for i in indices])
# print('产生的名字: ' , ''.join([ix_to_char[i] for i in indices]))

###################### 模块构建,包括梯度修剪&采样模块 ##########################

############################# 构建优化函数 ###################################
def optimize(X , Y , a_prev , parameters , learning_rate = 0.01):
    '''
    训练模型用的单步优化,全盘调用cllm_utils就完事
    Args:
        X: 整数列表
        Y: 整数列表,相当于 X索引左移一格
        a_prev: 其实只用给 a<0>
        parameters: 5个 W和 b
        learning_rate: 学习率

    Returns:
        loss: 损失函数
        gradients: 字典,包括 5个 dW,db
        a[len(X)-1]: 最后一个隐藏状态,(n_a , 1)
    '''
    # 前向传播
    loss , cache = cllm_utils.rnn_forward(X , Y , a_prev , parameters)
    # 反向传播
    gradients , a = cllm_utils.rnn_backward(X, Y, parameters, cache)
    # 梯度修剪,[-5 , 5]
    gradients = clip(gradients , 5)
    # 更新参数
    parameters = cllm_utils.update_parameters(parameters , gradients , learning_rate)
    return loss , gradients , a[len(X) - 1]

# print('=================测试optimize=================')
# np.random.seed(1)
# vocab_size, n_a = 27, 100
# a_prev = np.random.randn(n_a, 1)
# Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
# X = [12,3,5,11,22,3]
# Y = [4,14,11,22,25, 26]
#
# loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
# print("Loss =", loss)
# print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
# print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
# print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
# print("gradients[\"db\"][4] =", gradients["db"][4])
# print("gradients[\"dby\"][1] =", gradients["dby"][1])
# print("a_last[4] =", a_last[4])

############################# 构建优化函数 ###################################


############################### 训练模型 #####################################
'''
    创建样本的方法:
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]] 
        Y = X[1:] + [char_to_ix["\n"]]
'''
def model(data , ix_to_char , char_to_ix , num_iterations = 3500 , n_a = 50 , dino_names= 7 , vocab_size = 27):
    '''
    训练模型并且生成恐龙名字
    Args:
        data: 恐龙名字的训练集
        ix_to_char: 索引到字符的字典
        char_to_ix: 字符到索引的字典
        num_iterations: 迭代次数
        n_a: RNN单元数量
        dino_names: 每次迭代时采样的数量(生成7条名字)
        vocab_size: 出现过的所有字符数量

    Returns:
        parameters: 学习后的参数

    '''
    # 获取长度
    n_x = n_y = vocab_size
    # 初始化参数
    parameters = cllm_utils.initialize_parameters(n_a , n_x , n_y)
    # 初始化损失
    loss = cllm_utils.get_initial_loss(vocab_size , dino_names)
    # 构建恐龙名称列表,话说data完全没用吖...
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    # 打乱恐龙名字
    np.random.seed(0)
    np.random.shuffle(examples)
    # 初始化a0
    a_prev = np.zeros((n_a , 1))

    # 循环训练
    for j in range(num_iterations):
        # 创建样本
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]
        # 单步优化
        curr_loss , gradients , a_prev = optimize(X, Y, a_prev, parameters)
        # 让损失函数更平滑,加速训练
        loss = cllm_utils.smooth(loss , curr_loss)
        # 每2000次迭代进行一次采样
        if j % 2000 == 0:
            print('第%d次迭代,损失值为:%.2f' % (j+1 , loss))
            seed = 0
            for name in range(dino_names): # 产生7个例子看看
                sample_indices = sample(parameters , char_to_ix , seed)
                # 打印出采样结果
                cllm_utils.print_sample(sample_indices , ix_to_char)
                # 为了结果不一样
                seed += 1
            print('\n')
    return parameters

############################### 训练模型 #####################################

# 开始训练
start_time = time.clock()
parameters = model(data , ix_to_char , char_to_ix , num_iterations=20000)  # loss不收敛...
end_time = time.clock()
# 算时间
minium = end_time - start_time
print('执行了%d分%d秒' % (minium//60 , minium%60))