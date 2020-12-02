'''
1. 写一个迭代器，要求输出至今为止的所有闰年。如：
>>> leapYears = LeapYear()
>>> for i in leapYears:
        if i >=2000:
                print(i)
        else:
                break

2012
2008
2004
2000
提示：闰年判定法（(year%4 == 0 and year%100 != 0) or (year%400 == 0)）
'''

class LeapYear:
    def __init__(self):
        self.year = 2000

    def __iter__(self):
        return self
    def __next__(self):
        self.year += 1

        while 1 :
            if self.year >= 2019:
                raise StopIteration
            if (self.year%4 == 0 and self.year%100 != 0) or (self.year%400 == 0):
                break
            else:
                self.year += 1
            
        return self.year
            
        
leapYears = LeapYear()
for i in leapYears:
        if i >=2000:
                print(i)
        else:
                break
