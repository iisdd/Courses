import math

class Point():
    def __init__(self , x = 0 , y = 0):
        self.x = x
        self.y = y

    def getx(self):
        return self.x
    def gety(self):
        return self.y
    


class Line():
    def __init__(self , p1 , p2):
        self.x = p1.getx() - p2.getx()
        self.y = p1.gety() - p2.gety()
    def getLen(self):
        self.len = math.sqrt(self.x**2 + self.y**2)
        return self.len
