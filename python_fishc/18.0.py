'''
0. 编写一个符合以下要求的函数：
   
    a) 计算打印所有参数的和乘以基数（base=3）的结果
    b) 如果参数中最后一个参数为（base=5），则设定基数为5，基数不参与求和计算。
'''
def c(*args , base = 3):
    count = 0
    for each in args:
        count += each
    return count * base

print(c(1 , 2 , 3 ,base = 5))
print(c(1 , 2 , 3 ))
