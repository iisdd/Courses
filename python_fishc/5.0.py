'''
0. 针对视频中小甲鱼提到的小漏洞，再次改进我们的小游戏：当用户输入错误类型的时候，
及时提醒用户重新输入，防止程序崩溃。
#  方法1：
while 1:
    try:
        temp = int(input('请输入整数：'))
        break
    except ValueError:
        print('输入不合法！')
'''
#  方法2
while 1:
    temp = input('请输入整数：')
    if not temp.isdigit():
        print('输入不合法')
    else:
        temp = int(temp)
        break
