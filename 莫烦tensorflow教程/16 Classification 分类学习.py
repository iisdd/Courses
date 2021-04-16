# 通过分类手写数字来练习 tensorflow的分类
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 下载 & 读取数据
mnist = input_data.read_data_sets('MNIST_data' , one_hot = True)

# 定义添加层
def add_layer(inputs , in_size , out_size , activation_function = None,):
    Weights = tf.Variable(tf.random.normal([in_size , out_size]))
    biases = tf.Variable(tf.zeros([1 , out_size]) + 0.1,)                   # biases不能是0
    Wx_plus_b = tf.matmul(inputs , Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 计算准确度
def compute_accuracy(v_xs , v_ys):
    global y_pred
    y_pre = sess.run(y_pred , feed_dict={xs :v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre , 1) , tf.argmax(v_ys , 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))     # tf.cast:把数据类型转换成...
    result = sess.run(accuracy , feed_dict={xs : v_xs , ys : v_ys})
    return result

# 定义传入值 placeholder
xs = tf.placeholder(tf.float32 , [None , 28**2])                            # 不规定有多少个sample,规定输入的特征数量
ys = tf.placeholder(tf.float32 , [None , 10])

# 定义输出层
y_pred = add_layer(xs , 28**2 , 10 , activation_function=tf.nn.softmax)     # softmax一般用于多分类问题

# 定义loss为交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y_pred),reduction_indices=[1]))

# 训练
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 冲
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    b_x , b_y = mnist.train.next_batch(100) # 抽样100个,随机梯度下降
    sess.run(train_step , feed_dict={xs : b_x , ys : b_y})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images , mnist.test.labels
        )) # 用测试集检测准确度
