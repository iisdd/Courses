#   答案倒着来

import datetime as dt

class LeapYear:
    def __init__(self):
        self.now = dt.date.today().year

    def isLeapYear(self, year):
        if (year%4 == 0 and year%100 != 0) or (year%400 == 0):
            return True
        else:
            return False
        
    def __iter__(self):
        return self

    def __next__(self):
        while not self.isLeapYear(self.now):
            self.now -= 1 

        temp = self.now
        self.now -= 1
        
        return temp
leapYears = LeapYear()
for i in leapYears:
        if i >=2000:
                print(i)
        else:
                break
