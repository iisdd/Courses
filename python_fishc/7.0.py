'''
0. 视频中小甲鱼使用 if-elif-else 在大多数情况下效率要比全部使用 if 要高，
但根据一般的统计规律，一个班的成绩一般服从正态分布，
也就是说平均成绩一般集中在 70~80 分之间，因此根据统计规律，
我们还可以改进下程序以提高效率。
  
题目备忘：按照 100 分制，90 分以上成绩为 A，80 到 90 为 B，
60 到 80 为 C，60 以下为 D，写一个程序，当用户输入分数，
自动转换为ABCD 的形式打印。
'''
def jugde(num):
    if 60 <= num < 80:
        print('给你个C' )

    elif 80 <= num < 90:
        print('给你个B' )

    elif num < 60:
        print('给你个D' )

    else :
        print('给你个A' )


while 1:
    try:
        num = int(input('请输入分数：'))
        if 0 <= num <= 100:
            jugde(num)
            break
        else:
            print('那真的牛批！')

    except ValueError:
        print('爬！')
