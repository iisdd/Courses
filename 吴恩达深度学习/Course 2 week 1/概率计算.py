# import math
# p = math.log(50)/1000 # 分成10000份,模拟积分,时间步长0.01s

# # 1. 100s , 一枪不中
# res1 = (1-p)**10000
# print('res1 = ' ,res1)  #0.01849(100次) -> 0.0199846979403645(10000次)
#
# # 2. t1中一枪,剩下(200-2*t1)不中枪
# res2 = 0
# for t1 in range(1 , 10001):  # 不会一出门就中枪
#     res2 += (1-p)**(t1-1) * p * (1-p)**(20000-2*t1)
# print('res2 = ' ,res2) # 0.019592974605338057
#
# # 3. t1中一枪,过了t2中一枪,剩下(400-4*t1-2*t2)不中枪
# res3 = 0
# for t1 in range(1 , 10001):
#     for t2 in range(1 , 20001-2*t1):
#         res3 += (1-p)**(t1-1) * p * (1-p)**(t2-1) * p * (1-p)**(40000-4*t1-2*t2)
# print('res3 = ' ,res3)
#
# print('前三个结果和: ' , res1+res2+res3) # 0.05250897837971828
# 第四个单独算了


# 4. t1中一枪,过了t2中一枪,过了t3中一枪,剩下(800-8*t1-4*t2-2*t3)不中枪
# res4 = 0
# for t1 in range(1 , 1001):
#     for t2 in range(1 , 2001-2*t1):
#         for t3 in range(1 , 4001-4*t1-2*t2):
#             res4 += (1-p)**(t1-1) * p * (1-p)**(t2-1) * p * (1-p)**(t3-1) * p * (1-p)**(8000-8*t1-4*t2-2*t3)
# print('res4 : ' , res4)  # res4 :  0.007296854483242471(100)
# print(res1+res2+res3+res4)

# 结果0.05738,不对,小了点



# import math
# p = math.log(50)/10000
# print((1-p)**10000)  # 0.01998,原: 0.01849
#
# # 分的越细结果越精确

import numpy as np
from math import factorial
# lambd = np.log(50)/100
# # T分开讨论
#

def PMF(x,mu):
    res = 0
    for i in range(x):
        res += np.exp(-mu)*(mu**i)/factorial(i)
        print('x = %d'%i , res)
    return res
PMF(11 , 6)


#
# # 情况一: 一枪没中,走了100s
# mu1 = lambd*100
# res1 = PMF(0 , mu1)
#
# # 情况二: t1时刻中了一枪,又安全走了(200-2*t1)s
# res2 = 0
# for t1 in range(10000): # 分成100份来算？,模拟积分
#     mu1 = lambd*t1/100
#     mu2 = lambd*(200-2*t1/100)
#     res2 += PMF(1 , mu2) + PMF(0 , mu2)
# print(res2/10000)
