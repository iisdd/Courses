'''
    用Keras实现了莎士比亚诗歌生成器
'''
import time
#开始时间
start_time = time.clock()

from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from shakespeare_utils import *  # 这里就直接运行了
import sys
import io

#结束时间
end_time = time.clock()

#计算时差
minium = end_time - start_time

print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium%60)) + "秒")

# 训练一个epoch
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=10, callbacks=[print_callback])

# 运行此代码尝试不同的输入，而不必重新训练模型。
generate_output()