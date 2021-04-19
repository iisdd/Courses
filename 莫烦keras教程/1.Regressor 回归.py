"""
    用 keras做一个回归的例子:曲线拟合
    模型内容包括:
    1.创建模型 model = Sequential(), 可以一次全写完, 也可以用add一层一层搭
    2.搭积木 model.add()
    3.选择误差与优化器 model.compile()
    4.训练 model.train_on_batch(), model.fit()
    5.评估 cost, accuracy = model.evaluate()
    6.做预测 model.predict()
"""
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense                          # 选择需要用的网络结构,直接model.add
import matplotlib.pyplot as plt

# 创建数据集
X = np.linspace(-1 , 1 , 200)
np.random.shuffle(X)
noise = np.random.normal(0 , 0.05 , (200 , ))
Y = 0.5*X + 2 + noise
plt.scatter(X , Y)
plt.show()

X_train , Y_train = X[ : 160] , Y[ : 160]
X_test , Y_test = X[160 : ] , Y[160 : ]

# 像 pytorch一样一层一层叠上去就行
model = Sequential()

model.add(Dense(units=1 , input_dim=1))                 # 添加层:model.add
# 假如还要添加层的话,就不用加input_dim了,默认连接上一行的层
# model.add(Dense(units=1 ))

# 选择误差和优化器,compile:编译,model.compile
model.compile(loss = 'mse' , optimizer='sgd')

# training
print('=======training========')
for step in range(301):
    cost = model.train_on_batch(X_train , Y_train)      # 训练集的cost:model.train_on_batch
    if step % 50 == 0:
        print('train cost : ' , cost)

# testing
print('\n=======testing========')
cost = model.evaluate(X_test , Y_test , batch_size=40)  # 预测集的cost:model.evaluate
# batch_size=40,指抽出全部的预测集来算cost
print('test cost : ' , cost)
W , b = model.layers[0].get_weights()
print('Weights = ' , W , 'bias = ' , b)

# 画图
Y_pred = model.predict(X_test)  # 预测:model.predict
plt.scatter(X_test , Y_test)
plt.plot(X_test , Y_pred)
plt.show()
