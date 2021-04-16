# 用神经网络拟合一条曲线 y = 0.1*x + 0.3
import tensorflow as tf
import numpy as np

# 创造数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

###############神经网络结构####################
Weights = tf.Variable(tf.random.uniform([1] , -1.0 , 1.0))  # 一维随机数列,范围-1到1
biases = tf.Variable(tf.zeros([1]))                         # 一维全0数列 , 注意这个[1]要用()括起来

y_pred = Weights * x_data + biases
loss = tf.reduce_mean(tf.square(y_pred - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)          # 最原始的优化器 0.5 : lr
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()                        # 所有变量variables在神经网络中激活

##############################################

sess = tf.Session() # session:会话控制
sess.run(init) # 按下激活按钮

for step in range(201):
    if step % 20 == 0:
        print(step , sess.run(Weights) , sess.run(biases))
    sess.run(train)
        # sess.run(Weights)就是按一下 Weights这个按钮,查看它的值

