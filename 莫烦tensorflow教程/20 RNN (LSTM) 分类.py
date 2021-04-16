'''
    用手写数字的例子测试 LSTM的分类问题,把图片信息(28*28)拆成28条长度为28的数据送入RNN
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# 超参数
BATCH_SIZE = 64
TIME_STEP = 28  # 一共28行
INPUT_SIZE = 28 # 一行28个像素
LR = 0.01

# 读取数据
mnist = input_data.read_data_sets('./MNIST_data' , one_hot = True)
test_x = mnist.test.images[ : 2000]
test_y = mnist.test.labels[ : 2000]

# plot one example
# print(mnist.train.images.shape)     # (55000, 28 * 28)
# print(mnist.train.labels.shape)   # (55000, 10)
# plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
# plt.title('%i' % np.argmax(mnist.train.labels[0]))
# plt.show()

# 设置传入值
tf_x = tf.placeholder(tf.float32 , [None , TIME_STEP * INPUT_SIZE])     # shape : (batch , 784)
image = tf.reshape(tf_x , [-1 , TIME_STEP , INPUT_SIZE])                # shape : (m , 28 , 28)
tf_y = tf.placeholder(tf.int32 , [None , 10])                           # y的类型是整型

# RNN框架结构
#rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=64) # 更新了
rnn_cell = tf.keras.layers.LSTMCell(units=64)
outputs , (h_c , h_n) = tf.nn.dynamic_rnn(
    rnn_cell,  # 选择的 cell
    image,     # 输入
    initial_state=None,
    dtype = tf.float32,
    time_major=False,       # False:(batch , time step , input) ; True:(time step , batch , input)
)
output = tf.layers.dense(outputs[ : , - 1 , : ] , 10)   # 输出最后一行的结果,也就是看完了所有行之后再做判断

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y , logits = output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(                         # 自带的准确率计算BIF, 返回(acc , update_op)
    labels = tf.argmax(tf_y , axis = 1) , predictions=tf.argmax(output , axis = 1) , )[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init_op)

for step in range(1200):
    b_x , b_y = mnist.train.next_batch(BATCH_SIZE)
    _ , loss_ = sess.run([train_op , loss] , feed_dict={tf_x : b_x , tf_y : b_y})
    if step % 50 == 0:
        accuracy_ = sess.run(accuracy , {tf_x : b_x , tf_y : b_y})
        print('train loss : %.4f' % loss_ , '| test accuracy : %.2f' % accuracy_)



# 打印 10个预测例子
test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')

