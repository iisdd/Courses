"""
    用曲线拟合测试 LSTM 的回归问题
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Hyper Parameters
TIME_STEP = 10      # rnn time step,每次根据10个连着的数判断下一个数(验10推10)
INPUT_SIZE = 1      # rnn input size
CELL_SIZE = 32      # rnn cell size
LR = 0.02           # learning rate

# show data, 后面用不到x_np, y_np
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps); y_np = np.cos(steps)    # float32 for converting torch FloatTensor
plt.plot(steps, y_np, 'r-', label='target (cos)'); plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best'); plt.show()

# 设置传入值
tf_x = tf.placeholder(tf.float32 , [None , TIME_STEP , INPUT_SIZE])  # shape:(batch , 10 , 1)
tf_y = tf.placeholder(tf.float32 , [None , TIME_STEP , INPUT_SIZE])

# RNN部分
rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units = CELL_SIZE)
init_s = rnn_cell.zero_state(batch_size=1 , dtype = tf.float32)
outputs , final_s = tf.nn.dynamic_rnn(
    rnn_cell,                   # 选择cell
    tf_x,                       # 输入
    initial_state=init_s,       # 初始化hidden state
    time_major=False,           # shape : (batch , time step , input)
)
# 3D -> 2D(用于 Wx + b) -> 3D
outs2D = tf.reshape(outputs , [-1 , CELL_SIZE])
print(outs2D.shape)             # (10, 32)
# 把输出变成二维,然后接一个全连接层
net_outs2D = tf.layers.dense(outs2D , INPUT_SIZE)
print(net_outs2D.shape)         # (10, 1)
outs = tf.reshape(net_outs2D , [-1 , TIME_STEP , INPUT_SIZE]) # 再重新变回3D

loss = tf.losses.mean_squared_error(labels = tf_y , predictions=outs)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.figure(1 , figsize=(12 , 5))
plt.ion()

for step in range(60):
    start , end = step * np.pi , (step+1) * np.pi
    # 用sin预测cos
    steps = np.linspace(start , end , TIME_STEP)
    x = np.sin(steps)[np.newaxis , : ,np.newaxis]   # shape : (batch , time_step , input_size)
    y = np.cos(steps)[np.newaxis , : ,np.newaxis]
    # 循环训练state
    if 'final_s_' not in globals():
        feed_dict = {tf_x : x , tf_y : y}           # 最开始的init_s全0
    else:
        feed_dict = {tf_x : x , tf_y : y , init_s : final_s_}
    _ , pred_ , final_s_ = sess.run([train_op , outs , final_s] , feed_dict)

    # 画图部分
    plt.plot(steps, y.flatten(), 'r-'); plt.plot(steps, pred_.flatten(), 'b-')
    plt.ylim((-1.2, 1.2)); plt.draw(); plt.pause(0.05)

plt.ioff(); plt.show()
