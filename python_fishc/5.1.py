'''
1. 写一个程序，判断给定年份是否为闰年。（注意：请使用已学过的 BIF 进行灵活运用）

这样定义闰年的:能被4整除但不能被100整除,或者能被400整除都是闰年。
'''
def jugde(year):
    if ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0):
        print ('yes')
    else:
        print('nope')

while 1 :
    try:
        year = int(input('请输入年份：'))
        jugde(year)
        break
    except ValueError:
        print('请输入整数！')
        
