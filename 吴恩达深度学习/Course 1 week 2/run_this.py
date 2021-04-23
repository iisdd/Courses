"""
    课程一, 第二周作业,搭建一个识别猫的网络
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

# 用工具包导入数据
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# plt.imshow(train_set_x_orig[24])  # 看了下第25张图,是只橘猫,好肥
# plt.show()
m_train = train_set_y.shape[1]      # 训练集的图片数量
m_test = test_set_y.shape[1]        # 测试集数量
num_px = train_set_x_orig.shape[1]  # 图片的像素 : 64x64

# 现在看一下加载的数据具体情况
print('训练集的数量：m_train = ' , m_train)                  # 209
print('测试集的数量：m_test = ' , m_test)                    # 50
print('每张图片的宽&高：num_px = ' , num_px)                 # 64
print('每张图片的大小：(%d,%d,3)' % (num_px ,num_px))        # (64,64,3)
print('训练集图片的维度数：' + str(train_set_x_orig.shape))  # (209, 64, 64, 3)
print('训练集标签的维度数：' + str(train_set_y.shape))       # (1 , 209)
print('测试集图片的维度数：' + str(test_set_x_orig.shape))   # (50, 64, 64, 3)
print('测试集标签的维度数：' + str(test_set_y.shape))        # (1 , 50)

# 为了输入全连接,我们要把(64 , 64 , 3)压成(64*64*3 , 1)
# 先是训练集,降维然后转置
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0] , -1).T
# -1就是让电脑来填,保留原来的行数,列数为特征总数
# 然后是测试集
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0] , -1).T

print('\n训练集降维后的维度：' + str(train_set_x_flatten.shape))
print('训练集标签的维度：' + str(train_set_y.shape))
print('测试集降维后的维度：' + str(test_set_x_flatten.shape))
print('测试标签的维度：' + str(test_set_y.shape))

# 归一化处理数据
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmoid(z):
    # 连激活函数都要自己定义。。。忒原始了吧
    return 1/(1 + np.exp(-z))
# 测试sigmoid
# print('sigmoid(0) = ' , sigmoid(0))
# print('sigmoid(5) = %.2f 应该接近1'  %sigmoid(5))

def initialize_with_zeros(dim):
    # 为 w创建一个(dim , 1)的 0向量,同时把 b初始化成 0
    w = np.zeros(shape = (dim , 1))
    b = 0
    # 强行用一个老师讲的 assert
    assert(w.shape == (dim , 1))
    assert(isinstance(b , float) or isinstance(b , int))

    return(w , b)

def propagate(w , b , X , Y):
    # 实现前向和后向传播,返回 cost , dw , db
    m = X.shape[1]                      # 样本数
    # 正向传播部分
    A = sigmoid(np.dot(w.T , X) + b)    # b就是个数,python里有 broadcasting可以自动补全,A就是y估计
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    # cost越小越准

    # 反向传播部分
    dw = (1 / m) * np.dot(X , (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    # 用 assert确保数据格式正确
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost) # 压成一个数,没有维度
    assert(cost.shape == ())

    # 用一个字典把 dw , db 存起来
    grads = {
        'dw':dw,
        'db':db,
    }
    return (grads , cost)

# 用梯度下降来优化参数
def optimize(w , b , X , Y , num_iterations , learning_rate , print_cost = False):
    # print_cost -  每 100步打印一次cost
    # 返回 params - 包括 w 和 b 的字典 , grads - 包括 dw 和 db的字典 ,costs - 用于绘制成本曲线的列表
    # 循环包括两个部分 1、使用propagate()计算成本和梯度。 2、使用 w 和 b的梯度下降法更新参数
    costs = []
    for i in range(num_iterations):
        grads , cost = propagate(w , b , X , Y)

        dw = grads['dw']
        db = grads['db']
        # 更新参数,梯度是沿着最大化cost的方向,所以要减去 dw、db
        w -= learning_rate * dw
        b -= learning_rate * db

        # 100步记录一下成本
        if i % 100 == 0:
            costs.append(cost)
            #print('迭代的次数：%d , 误差值：%.2f' %(i , cost))

    params = {
        'w':w,
        'b':b,
    }
    grads = {
        'dw':dw,
        'db':db,
    }
    return (params , grads , costs)

def predict(w , b , X):
    # 把 X喂入网络输出 Y_prediction , shape:(1 , m)
    m = X.shape[1]
    Y_prediction = np.zeros((1 , m))
    w = w.reshape(X.shape[0] , 1)

    A = sigmoid(np.dot(w.T , X) + b)
    for i in range(m): # 用round误差就不对了
        Y_prediction[0 , i] = 1 if A[0 , i] > 0.5 else 0
    # Y_prediction = np.round(Y_prediction)
    assert(Y_prediction.shape == (1 , m))
    return Y_prediction


def run(X_train , Y_train , X_test , Y_test , num_iterations = 2000 , learning_rate = 0.5 , print_cost = False):
    # 通过调用上面定义的方法,返回一个包括模型信息的字典d
    # 初始化
    w , b = initialize_with_zeros(X_train.shape[0])
    # 训练 w , b
    parameters , _ , costs = optimize(w , b , X_train , Y_train , num_iterations , learning_rate , print_cost)
    # 梯度什么的不关键,用_代替就行
    # 从字典中拉出 w , b
    w , b = parameters['w'] , parameters['b']

    Y_prediction_test = predict(w , b , X_test)
    Y_prediction_train = predict(w , b , X_train)

    # 打印预测后的准确性
    print('训练集准确性：' , format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100) , '%')
    print('测试集准确性：' , format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) , '%')

    d = {
        'costs' : costs,
        'Y_prediction_test' : Y_prediction_test,
        'Y_prediction_train' : Y_prediction_train,
        'w' : w,
        'b' : b,
        'learning_rate' : learning_rate,
        'num_iterations' : num_iterations,
    }
    return d

d = run(train_set_x , train_set_y , test_set_x , test_set_y , num_iterations=2000 , learning_rate=0.005 , print_cost=True)

# 画图部分
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iteration(per hundreds)')
plt.title('Learning rate = ' + str(d['learning_rate']))
plt.show()

# 看下不同 lr的学习效果
learning_rates = [0.01, 0.001, 0.0001]
models = {} # 用字典存不同的lr的d
for i in learning_rates:
    print('\n' + "-------------------------------------------------------" + '\n')
    print ("learning rate is: " + str(i))
    models[str(i)] = run(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = i, print_cost = False)


for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()