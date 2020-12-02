'''
1. 尝试写代码实现以下截图功能：
>>>
请输入一个整数：5
1
2
3
4
5
'''
def printnum():
    try:
        num = int(input('请输入一个整数：'))
        while num:
            print(num)
            num -= 1
    except ValueError:
        print('请输入整数，憨憨')
        printnum()

printnum()
