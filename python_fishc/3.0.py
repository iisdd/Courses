def calc():
    '''
0. 还记得我们第一讲的动动手的题目吗？这一次要求使用变量，计算一年有多少秒？

提示：可以以 DaysPerYear（每年天数），HoursPerDay（每天小时数），
MinutesPerHour（每小时分钟数），SecondsPerMinute（每分钟秒数）为变量名。
'''
    a = DaysPerYear = 365
    b = HoursPerDay = 24
    c = MinutesPerHour = 60
    d = SecondsPerMinute = 60
    result = a * b * c * d
    print(result)

if __name__ == '__main__':
    calc()
