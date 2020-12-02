'''
0. 编写一个进制转换程序，程序演示如下（提示，十进制转换二进制可以用bin()这个BIF）：
'''
while 1:
    temp = input('请输入一个整数(输入Q结束程序)：')
    if temp == 'Q':
        break
    temp = int(temp)
    print('十进制 -> 十六进制：0x%x' % temp)
    print('十进制 -> 八进制：0o%o' % temp)
    print('十进制 -> 二进制：%s' %bin(temp))
