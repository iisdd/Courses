'''
0. 使用递归编写一个十进制转换为二进制的函数（要求采用“取2取余”的方式，
结果与调用bin()一样返回字符串形式）。
'''

def change(n):
    str1 = ''
    if n < 2:
        return str(n)

    else:

        return change(n // 2) + str(n % 2) 
    



print('0b' + change(4))
