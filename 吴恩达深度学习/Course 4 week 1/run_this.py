# 试一下加个模型保存
import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)



# arr3D = np.array([[[1, 1, 2, 2, 3, 4],
#                    [1, 1, 2, 2, 3, 4],
#                    [1, 1, 2, 2, 3, 4]],
#
#                   [[0, 1, 2, 3, 4, 5],
#                    [0, 1, 2, 3, 4, 5],
#                    [0, 1, 2, 3, 4, 5]],
#
#                   [[1, 1, 2, 2, 3, 4],
#                    [1, 1, 2, 2, 3, 4],
#                    [1, 1, 2, 2, 3, 4]]])
#
# print ('constant:  \n' + str(np.pad(arr3D, ((0, 0), (1, 1),(2,2)), 'constant' , constant_values=((1,2)))))
# (0,0):前面后面加几个0矩阵 , (1,1):头尾加几行0 , (2,2):左右加几列0, constant_values=((1,2)):前/头/左补1,后/尾/右补2

# 先定义一个padding
def zero_pad(X , pad): # X.shape = [n , n_h , n_w , n_c]
    X_paded = np.pad(X , (
        (0,0), # 样本数不补
        (pad,pad), # 高度
        (pad,pad), # 宽度
        (0,0), # 通道数
    ),
        'constant',constant_values=0)
    return X_paded

# # 测试padding
# np.random.seed(1)
# x = np.random.randn(4,3,3,2)
# x_paded = zero_pad(x,2)
# #查看信息
# print ("x.shape =", x.shape)
# print ("x_paded.shape =", x_paded.shape)
# print ("x[1, 1] =", x[1, 1])
# print ("x_paded[1, 1] =", x_paded[1, 1])
#
# #绘制图
# fig , axarr = plt.subplots(1,2)  #一行两列
# axarr[0].set_title('x')
# axarr[0].imshow(x[0,:,:,0])
# axarr[1].set_title('x_paded')
# axarr[1].imshow(x_paded[0,:,:,0])
# plt.show()

def conv_single_step(a_slice_prev , W , b): # 对filter大小的切块进行卷积
    s = np.sum(np.multiply(a_slice_prev , W)) + b
    return s

# # 测试
# np.random.seed(1)
#
# #这里切片大小和过滤器大小相同
# a_slice_prev = np.random.randn(4,4,3)
# W = np.random.randn(4,4,3)
# b = np.random.randn(1,1,1)
#
# Z = conv_single_step(a_slice_prev,W,b)
#
# print("Z = " + str(Z))

def conv_forward(A_prev, W, b, hparameters): # 卷积的前向传播
    # 上一层的输出
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # 权重矩阵
    (f, f, n_C_prev, n_C) = W.shape
    # 获取超参
    stride = hparameters['stride']
    pad = hparameters['pad']
    # 理论上卷积后的高度和宽度
    n_H = int((n_H_prev - f + 2*pad)/stride) + 1
    n_W = int((n_W_prev - f + 2*pad)/stride) + 1

    # 初始化卷积输出Z
    Z = np.zeros((m, n_H, n_W, n_C))
    # 对上一层输出进行填充
    A_prev_pad = zero_pad(A_prev , pad)

    for i in range(m):  # 遍历样本
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                    # 切片
                    a_slice_prev = a_prev_pad[vert_start:vert_end , horiz_start:horiz_end , : ]
                    # 单步卷积
                    Z[i,h,w,c] = conv_single_step(a_slice_prev , W[: , : , : , c] , b[0,0,0,c])
                    # 结果的第c层,每次卷积的filter厚度都是n_C_prev

    # 验证输出shape
    assert(Z.shape == (m , n_H , n_W , n_C))
    # 缓存用于反向传播
    cache = (A_prev , W , b , hparameters)
    return (Z , cache)


# # 测试
# np.random.seed(1)
#
# A_prev = np.random.randn(10,4,4,3)
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(1,1,1,8)
#
# hparameters = {"pad" : 2, "stride": 1}
#
# Z , cache_conv = conv_forward(A_prev,W,b,hparameters)
#
# print("np.mean(Z) = ", np.mean(Z))
# print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])


def pool_forward(A_prev, hparameters, mode='max'):
    # 池化层前向传播
    # 获取输入数据信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # 获取超参数信息
    f = hparameters['f']
    stride = hparameters['stride']

    # 计算理论输出维度
    n_H = int((n_H_prev - f)/stride) + 1
    n_W = int((n_W_prev - f)/stride) + 1
    n_C = n_C_prev
    # 初始化输出矩阵
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                # 切片
                    a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    # 对切片进行池化操作
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_slice_prev)

    # 检验数据结构
    assert (A.shape == (m, n_H, n_W, n_C))
    # 储存反向传播的值
    cache = (A_prev, hparameters)

    return A, cache

np.random.seed(1)
A_prev = np.random.randn(2,4,4,3)
hparameters = {"f":4 , "stride":1}

A , cache = pool_forward(A_prev,hparameters,mode="max")
A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A =", A)
print("----------------------------")
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A =", A)