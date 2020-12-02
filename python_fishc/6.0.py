'''
0. 请写一个程序打印出 0~100 所有的奇数。
'''
def print_odd():
    try:
        start = int(input('请输入起始点：'))
        end = int(input('请输入终点：'))

    except ValueError:
        print('请输入整数！')
        print_odd(start , end)

    for each in range(start , end + 1 ):
        if each % 2:
            print(each , end = ' ')


print_odd()
