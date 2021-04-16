# 这一节介绍 tensorflow的两种打开模式
import tensorflow as tf

matrix1 = tf.constant([[3 , 3]])
matrix2 = tf.constant([[2],
                      [2]])

product = tf.matmul(matrix1 , matrix2)  # 相当于 np.dot()
# print(product) # 这不能打印出数字嗷

# 方法一
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

# 方法二,尊贵的 with
with tf.Session() as sess: # 自动关闭 : sess.close()
    result2 = sess.run(product)
    print(result2)
# 这里可以打别的巴拉巴拉,上面那个会话就已经关闭了


