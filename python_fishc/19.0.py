'''
0. 编写一个函数，判断传入的字符串参数是否为“回文联”
（回文联即用回文形式写成的对联，既可顺读，也可倒读。例如：上海自来水来自海上）
'''
def ishuiwen():
    test = input('请输入一句话：')
    length = len(test) // 2
    flag = 1
    for each in range(length):
        if test[each] != test[-each - 1]:
            flag = 0
    if flag == 1:
        print('是回文联！')
    else:
        print('不是回文联！')
    
ishuiwen()
