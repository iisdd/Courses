'''
1. 编写程序：calc.py 要求用户输入1到100之间数字并判断，
输入符合要求打印“你妹好漂亮”，不符合要求则打印“你大爷好丑”
'''
def calc():
    try:
        temp = int(input('清输入1到100之间的数字：'))
        if (temp > 1) and (temp < 100):
            print('天才！')
        else:
            print('会数数不？')

    except ValueError:
        print('憨批！')



if __name__ == '__main__':
    calc()
