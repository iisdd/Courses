# 这节介绍 placeholder , 规定一个待传入的值 , 要输入的时候再用feed_dict={}的形式喂入数据
import tensorflow as tf

#input1 = tf.placeholder(tf.float32 , [2 , 2]) # 2行2列规定结构
input1 = tf.placeholder(tf.float32)                                         # placeholder里面要填(数据类型 , 形状)
input2 = tf.placeholder(tf.float32)

output = tf.matmul(input1 , input2)

with tf.Session() as sess:
    print(sess.run(output , feed_dict={input1:[[7.]] , input2:[[2.]]}))     # 记得打两个括号形成矩阵
    # sess.run(结构  , feed_dict = {喂入数据})
