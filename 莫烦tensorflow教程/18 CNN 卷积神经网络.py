'''
    用手写数字辨识的例子来练习一下 CNN
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001

mnist = input_data.read_data_sets('MNIST_data' , one_hot = True)
test_x = mnist.test.images[ : 2000]
test_y = mnist.test.labels[ : 2000]

print(mnist.train.images.shape)  # (55000 , 28 * 28)
print(mnist.train.labels.shape)  # (55000 , 10)

# 画出一个例子
plt.imshow(mnist.train.images[0].reshape((28 , 28)) , cmap='gray')  # reshape一定要用()包起来
plt.title('%i' % np.argmax(mnist.train.labels[0]))
plt.show()

# 设置传入值
tf_x = tf.placeholder(tf.float32 , [None , 28*28]) / 255.
image = tf.reshape(tf_x , [-1 , 28 , 28 , 1])       # shape:(batch , height , width , channel)
# tf_x表示 image的数据类型和 tf_x一样
tf_y = tf.placeholder(tf.float32 , [None , 10])


# 神经网络部分: 卷积 - 池化 - 卷积 - 池化 - 全连接 - loss - train_op
# CNN部分,一层一层叠
conv1 = tf.layers.conv2d(                           # shape:(28 , 28 , 1)
    inputs = image,
    filters = 16,                                   # 卷积核数量
    kernel_size = 5,                                # 卷积核尺寸
    strides = 1,                                    # 步长
    padding = 'same',
    # 填充方式,有两种选择,same和 valid,总归看下来就是same抽样的特征和原来一样大,valid会比原来小一点
    activation = tf.nn.relu,
)   # -> (28 , 28 , 16)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,                                    # 选择每一个2x2格子里的最大值
    strides=2,
)
conv2 = tf.layers.conv2d(pool1 , 32 , 5 , 1 , 'same' , activation=tf.nn.relu)   # ->(14 , 14 , 32)
pool2 = tf.layers.max_pooling2d(conv2 , 2 , 2)                                  # -> (7 , 7 , 32)
flat = tf.reshape(pool2 , [-1 , 7 * 7 * 32])                                    # 数据类型和 pool2一样,输出shape变成一维了:(7*7*32 , )前面-1指样本数 m
output = tf.layers.dense(flat , 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y , logits = output)
# 交叉熵 loss = -(y * log(y_pred) + (1 - y) * log(1 - y_pred))
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(  # 这个BIF返回 (acc , update_op)
    labels = tf.argmax(tf_y , axis = 1) , predictions=tf.argmax(output , axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer()) # 后面的local是为了 accuracy_op
sess.run(init_op)

# 画图部分
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('\nPlease install sklearn for layer visualization\n')
def plot_with_labels(lowDWeights, labels):
    plt.cla(); X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()

# 训练部分
for step in range(600):
    b_x , b_y = mnist.train.next_batch(BATCH_SIZE)  # 全自动抽样,我哭辣
    _ , loss_ = sess.run([train_op , loss] , feed_dict={tf_x : b_x , tf_y : b_y})
    # 为了区分网络和输出,一般在输出后面加个 '_'
    if step % 50 == 0:  # 每50步丢入测试集试一下loss
        accuracy_ , flat_representation = sess.run([accuracy , flat] , feed_dict={tf_x : test_x , tf_y : test_y})
        print('Step: ' , step , '| train loss: %.4f' % loss_ , '| test accuracy: %.2f' % accuracy_)

        if HAS_SK:
            # Visualization of trained flatten layer (T-SNE)
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); plot_only = 500
            low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
            labels = np.argmax(test_y, axis=1)[:plot_only]; plot_with_labels(low_dim_embs, labels)
plt.ioff()

# 打印 10个测试集的预测值
test_output = sess.run(output , {tf_x : test_x[ : 10]})
pred_y = np.argmax(test_output , 1)
print(pred_y , '预测值')
print(np.argmax(test_y[ : 10] , 1) , '实际值')

