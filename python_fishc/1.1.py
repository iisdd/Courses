'''
1. 在交互模式中，使用 Python 计算一年有多少秒？
>>>print( 365 * 24 * 60 * 60 )
>>>31536000
'''
def calc():
    a = days_per_year = 365
    b = hours_per_day = 24
    c = mins_per_hour = 60
    d = secs_per_min = 60
    result = a * b * c * d
    return result
