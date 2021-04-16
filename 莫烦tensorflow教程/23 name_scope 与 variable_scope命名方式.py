'''
本节包括两种定义scope的方式,加一个RNN的例子
两种创建变量的方式 : tf.get_variable , tf.Variable , 使用 tf.get_variable()定义的变量不会被 tf.name_scope()当中的名字所影响.
在同一个scope中命名同样名字的变量会自动在名字后面加 '_1' , '_2'
'''
import tensorflow as tf

with tf.name_scope('a_name_scope'):
    initializer = tf.constant_initializer(value = 1)
    var1 = tf.get_variable(name = 'var1' , shape=[1] , dtype=tf.float32 , initializer=initializer)
    var2 = tf.Variable(name = 'var2' , initial_value=[2] , dtype=tf.float32)
    var21 = tf.Variable(name = 'var2' , initial_value=[2.1] , dtype=tf.float32)
    var22 = tf.Variable(name = 'var2' , initial_value=[2.2] , dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name)        # var1:0
    print(sess.run(var1))   # [1.]
    print(var2.name)        # a_name_scope/var2:0
    print(sess.run(var2))   # [2.]
    print(var21.name)       # a_name_scope/var2_1:0
    print(sess.run(var21))  # [2.1]
    print(var22.name)       # a_name_scope/var2_2:0
    print(sess.run(var22))  # [2.2]

'''
如果想要达到重复利用变量的效果, 我们就要使用 tf.variable_scope(), 
并搭配 tf.get_variable() 这种方式产生和提取变量. 不像 tf.Variable() 每次都会产生新的变量,
tf.get_variable() 如果遇到了同样名字的变量时, 它会单纯的提取这个同样名字的变量(避免产生新变量). 而在重复使用的时候,
一定要在代码中强调 scope.reuse_variables(), 否则系统将会报错, 以为你只是单纯的不小心重复使用到了一个变量.
'''
with tf.variable_scope('a_variable_scope') as scope:
    initializer = tf.constant_initializer(value = 3)
    var3 = tf.get_variable(name = 'var3' , shape=[1] , dtype=tf.float32 , initializer=initializer)
    scope.reuse_variables()   # 有这句话就不会产生新变量,而是提取这个变量,不加这句话会报错,以为你调用了一个已经有的变量
    var3_reuse = tf.get_variable(name = 'var3' , )
    var4 = tf.Variable(name = 'var4' , initial_value = [4] , dtype=tf.float32)
    var4_reuse = tf.Variable(name = 'var4' , initial_value=[4] , dtype = tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var3.name)            # a_variable_scope/var3:0
    print(sess.run(var3))       # [3.]
    print(var3_reuse.name)      # a_variable_scope/var3:0 , 这两个名字就是一样的
    print(sess.run(var3_reuse)) # [3.]
    print(var4.name)            # a_variable_scope/var4:0
    print(sess.run(var4))       # [4.]
    print(var4_reuse.name)      # a_variable_scope/var4_1:0 , 这就不一样了新创建了一个变量
    print(sess.run(var4_reuse)) # [4.]


# RNN使用例子,就是训练和测试用的步长不同,但测试的时候也需要调用训练好的参数


class TrainConfig:
    batch_size = 20
    time_steps = 20
    input_size = 10
    output_size = 2
    cell_size = 11
    learning_rate = 0.01


class TestConfig(TrainConfig):
    time_steps = 1


class RNN(object):

    def __init__(self, config):
        self._batch_size = config.batch_size
        self._time_steps = config.time_steps
        self._input_size = config.input_size
        self._output_size = config.output_size
        self._cell_size = config.cell_size
        self._lr = config.learning_rate
        self._built_RNN()

    def _built_RNN(self):
        with tf.variable_scope('inputs'):
            self._xs = tf.placeholder(tf.float32, [self._batch_size, self._time_steps, self._input_size], name='xs')
            self._ys = tf.placeholder(tf.float32, [self._batch_size, self._time_steps, self._output_size], name='ys')
        with tf.name_scope('RNN'):
            with tf.variable_scope('input_layer'):
                l_in_x = tf.reshape(self._xs, [-1, self._input_size], name='2_2D')  # (batch*n_step, in_size)
                # Ws (in_size, cell_size)
                Wi = self._weight_variable([self._input_size, self._cell_size])
                print(Wi.name)
                # bs (cell_size, )
                bi = self._bias_variable([self._cell_size, ])
                # l_in_y = (batch * n_steps, cell_size)
                with tf.name_scope('Wx_plus_b'):
                    l_in_y = tf.matmul(l_in_x, Wi) + bi
                l_in_y = tf.reshape(l_in_y, [-1, self._time_steps, self._cell_size], name='2_3D')

            with tf.variable_scope('cell'):
                cell = tf.contrib.rnn.BasicLSTMCell(self._cell_size)
                with tf.name_scope('initial_state'):
                    self._cell_initial_state = cell.zero_state(self._batch_size, dtype=tf.float32)

                self.cell_outputs = []
                cell_state = self._cell_initial_state
                for t in range(self._time_steps):
                    if t > 0: tf.get_variable_scope().reuse_variables()
                    cell_output, cell_state = cell(l_in_y[:, t, :], cell_state)
                    self.cell_outputs.append(cell_output)
                self._cell_final_state = cell_state

            with tf.variable_scope('output_layer'):
                # cell_outputs_reshaped (BATCH*TIME_STEP, CELL_SIZE)
                cell_outputs_reshaped = tf.reshape(tf.concat(self.cell_outputs, 1), [-1, self._cell_size])
                Wo = self._weight_variable((self._cell_size, self._output_size))
                bo = self._bias_variable((self._output_size,))
                product = tf.matmul(cell_outputs_reshaped, Wo) + bo
                # _pred shape (batch*time_step, output_size)
                self._pred = tf.nn.relu(product)    # for displacement

        with tf.name_scope('cost'):
            _pred = tf.reshape(self._pred, [self._batch_size, self._time_steps, self._output_size])
            mse = self.ms_error(_pred, self._ys)
            mse_ave_across_batch = tf.reduce_mean(mse, 0)
            mse_sum_across_time = tf.reduce_sum(mse_ave_across_batch, 0)
            self._cost = mse_sum_across_time
            self._cost_ave_time = self._cost / self._time_steps

        with tf.variable_scope('trian'):
            self._lr = tf.convert_to_tensor(self._lr)
            self.train_op = tf.train.AdamOptimizer(self._lr).minimize(self._cost)

    @staticmethod
    def ms_error(y_target, y_pre):
        return tf.square(tf.subtract(y_target, y_pre))

    @staticmethod
    def _weight_variable(shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=0.5, )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    @staticmethod
    def _bias_variable(shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    train_config = TrainConfig()
    test_config = TestConfig()

    # the wrong method to reuse parameters in train rnn
    # 开了两个scope,并没有共享参数
    with tf.variable_scope('train_rnn'):
        train_rnn1 = RNN(train_config)
    with tf.variable_scope('test_rnn'):
        test_rnn1 = RNN(test_config)

    # the right method to reuse parameters in train rnn
    # 同用一个scope,两个变量共享参数
    with tf.variable_scope('rnn') as scope:
        sess = tf.Session()
        train_rnn2 = RNN(train_config)
        scope.reuse_variables()
        test_rnn2 = RNN(test_config)
        # tf.initialize_all_variables() no long valid from
        # 2017-03-02 if using tensorflow >= 0.12
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

