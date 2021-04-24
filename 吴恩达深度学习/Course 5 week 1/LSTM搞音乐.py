from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import IPython
import sys
from music21 import *
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *

X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)  # (60 , 30 , 78): 60个样本,30个时间步,78长度的one-hot向量
# 从一段较长的音乐中抽30个值出来训练
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)  # (30 , 60 , 78): X是(m , Tx , 78) , Y是(Ty , m , 78),为了list(Y)产生长度为30的列表与y_hat求loss

# 构建模型
n_a = 64
reshapor = Reshape((1 , n_values))  # 变成one-hot形式
LSTM_cell = LSTM(n_a , return_state=True)
densor = Dense(n_values , activation='softmax')


# 创建模型实体
def djmodel(Tx , n_a , n_values):
    '''
    Args:
        Tx: 语料库的长度(时间步)
        n_a: LSTM激活神经元的数量
        n_values: 音乐数据中能用 one-hot表示的种类数量

    Returns:
        model: 神经网络实体架构

    '''
    # 定义输入数据维度
    X = Input((Tx , n_values))
    # 初始化隐藏状态和记忆状态
    a0 = Input(shape = (n_a , ) , name = 'a0')
    c0 = Input(shape = (n_a , ) , name = 'c0')
    a = a0
    c = c0

    # 1.创建一个列表存放LSTM所有时间步的输出
    outputs = []
    # 2.循环
    for t in range(Tx):
        # 2.A.拿出第t个时间步的X
        x = Lambda(lambda x : X[ : , t , : ])(X)
        # 2.B.对x重构
        x = reshapor(x)
        # 2.C.单步传播,更新a,c
        a , _ , c = LSTM_cell(x , initial_state=[a , c])
        # 2.D.softmax输出
        out = densor(a)
        # 存下来输出结果
        outputs.append(out)
    # 3.自定义模型实体
    model = Model(inputs = [X, a0, c0] , outputs = outputs)
    return model

# 获取模型,编译
model = djmodel(Tx = 30 , n_a = 64, n_values = 78)
opt = Adam(lr = 0.01 , beta_1=0.9 , beta_2=0.999 , decay=0.01)
model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
# 初始化a0,c0
m = 60
a0 = np.zeros((m , n_a))
c0 = np.zeros((m , n_a))

# 拟合模型
import time

start_time = time.clock()

# 开始拟合
model.fit([X , a0 , c0] , list(Y) , epochs = 100)
# Y:长度为30的列表,每个元素shape:(60 , 78)

end_time = time.clock()
minium = end_time - start_time
print('执行了%d分,%d秒' % (minium//60 , minium%60))


# 用训练好的model生成音乐
def music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=100):
    """
    参数：
        LSTM_cell -- 来自model()的训练过后的LSTM单元，是keras层对象。
        densor -- 来自model()的训练过后的"densor"，是keras层对象
        n_values -- 整数，唯一值的数量
        n_a -- LSTM单元的数量
        Ty -- 整数，生成的是时间步的数量

    返回：
        inference_model -- Kears模型实体
    """

    # 定义模型输入的维度
    x0 = Input(shape=(1, n_values))

    # 定义s0，初始化隐藏状态
    a0 = Input(shape=(n_a,), name="a0")
    c0 = Input(shape=(n_a,), name="c0")
    a = a0
    c = c0
    x = x0

    # 步骤1：创建一个空的outputs列表来保存预测值。
    outputs = []

    # 步骤2：遍历Ty，生成所有时间步的输出
    for t in range(Ty):
        # 步骤2.A：在LSTM中单步传播
        a, _, c = LSTM_cell(x, initial_state=[a, c])

        # 步骤2.B：使用densor()应用于LSTM_Cell的隐藏状态输出
        out = densor(a)

        # 步骤2.C：预测值添加到"outputs"列表中
        outputs.append(out)

        # 根据“out”选择下一个值，并将“x”设置为所选值的一个独热编码，
        # 该值将在下一步作为输入传递给LSTM_cell。我们已经提供了执行此操作所需的代码
        x = Lambda(one_hot)(out) # 生成一个喂给下一个

    # 创建模型实体
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs) # 默认x0,a0,c0全是0向量

    return inference_model

# 获取模型实体，模型被硬编码以产生50个值
inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)

#创建用于初始化x和LSTM状态变量a和c的零向量。
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))


# 预测与采样
def predict_and_sample(inference_model, x_initializer=x_initializer, a_initializer=a_initializer,
                       c_initializer=c_initializer):
    """
    使用模型预测当前值的下一个值。

    参数：
        inference_model -- keras的实体模型
        x_initializer -- 初始化的独热编码，维度为(1, 1, 78)
        a_initializer -- LSTM单元的隐藏状态初始化，维度为(1, n_a)
        c_initializer -- LSTM单元的状态初始化，维度为(1, n_a)

    返回：
        results -- 生成值的独热编码向量，维度为(Ty, 78)
        indices -- 所生成值的索引矩阵，维度为(Ty, 1)
    """
    # 步骤1：模型来预测给定x_initializer, a_initializer and c_initializer的输出序列
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])

    # 步骤2：将“pred”转换为具有最大概率的索引数组np.array()。
    indices = np.argmax(pred, axis=-1)

    # 步骤3：将索引转换为它们的一个独热编码。
    results = to_categorical(indices, num_classes=78)

    return results, indices


results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))

out_stream = generate_music(inference_model)