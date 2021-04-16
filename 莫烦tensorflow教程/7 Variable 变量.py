# 搞一个计数的变量counter
import tensorflow as tf
# 经典语录：它必须定义它是个变量,它才是个变量...
state = tf.Variable(0 , name = 'counter')
# print(state.name) # 输出counter:0 因为是第一个变量所以是0
one = tf.constant(1)

new_value = tf.add(state , one)
update = tf.assign(state , new_value)   #  把 new_value 赋给 state

init = tf.initialize_all_variables()    # 在tensorflow中激活所有变量
# init很关键的嗷

with tf.Session() as sess:
    sess.run(init) # 必有的开头！！！
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
