# 使用 tf.train.Saver()来保存提取
# 保存:
# saver = tf.train.Saver()
# saver.save(sess , 路径)
# 加载:
# saver.restore(sess , 路径)

import tensorflow as tf
import numpy as np

# # 保存权值到文件
# W = tf.Variable([[1 , 2 , 3 ],
#                  [3 , 4 , 5 ]],dtype = tf.float32,name = 'weights')  # 数据类型大多用float32
# b = tf.Variable([[1 , 2 , 3]] , dtype = tf.float32 , name = 'biases')
#
# init = tf.initialize_all_variables()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess , 'my_net/save_net.ckpt')
#     print('Save to path: ' , save_path)

# 提取变量
# 必须重新定义一个框架,框架的 shape和 type要和被提取的一样
W = tf.Variable(np.arange(6).reshape((2 , 3)) , dtype=tf.float32 , name = 'weights')
b = tf.Variable(np.arange(3).reshape((1 , 3)) , dtype=tf.float32 , name = 'biases')

# 不需要initial
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess , 'my_net/save_net.ckpt')  # restore : 归还,复原
    print('weights : ' , sess.run(W))
    print('biases : ' , sess.run(b))

