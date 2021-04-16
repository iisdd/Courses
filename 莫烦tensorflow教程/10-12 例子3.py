# 建造第一个神经网络 包括：1、添加层 def add_layer()  2、建造神经网络  3、结果可视化
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs , in_size , out_size , activation_function = None):
    Weights = tf.Variable(tf.random.normal([in_size , out_size])) # 大写指矩阵,形状后面还可以接上下限
    biases = tf.Variable(tf.zeros([1 , out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs , Weights) + biases
    # inputs.shape = (m , in_size) , Weights.shape = (in_size , out_size)
    # 乘出来shape就是 (m , out_size)
    # 激活部分
    if activation_function is None:
        outputs = Wx_plus_b # 线性关系
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 创建假数据
x_data = np.linspace(-1 , 1 , 300)[ : , np.newaxis]
noise = np.random.normal(0 , 0.05 , x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
# 创建传入值,等下用来填feed_dict
xs = tf.placeholder(tf.float32 , [None , 1])
ys = tf.placeholder(tf.float32 , [None , 1])

l1 = add_layer(xs , 1 , 10 , activation_function=tf.nn.relu)        # 输入层
# l1 是这个layer的输出值(outputs)
prediction = add_layer(l1 , 10 , 1 , activation_function=None)      # 输出层

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))                        # 均方差
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

# 开始启动
sess = tf.Session()
sess.run(init)

# 可视化部分
fig = plt.figure()      # 生成一个图片框
ax = fig.add_subplot(1 , 1 , 1)
ax.scatter(x_data , y_data)
plt.ion()               # plot后不暂停,继续走程序
plt.show()


for i in range(1001):
    sess.run(train_step , feed_dict={xs : x_data , ys : y_data})
    if i % 50 == 0:                                                 # 相当于描 21 根线
        print('迭代次数：' , i , 'loss : '  , sess.run(loss , feed_dict={xs : x_data , ys : y_data}))
        try:
            ax.lines.remove(lines[0])                               # 抹除前一条线,第一次会报错
        except Exception:
            pass
        prediction_value = sess.run(prediction , feed_dict={xs : x_data})
        lines = ax.plot(x_data , prediction_value , 'r-' , lw = 5)  # 红线,线宽为5
        plt.pause(0.5)
plt.ioff()
