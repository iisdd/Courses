'''练习要求：先定义一个温度类，然后定义两个描述符类用于描述摄氏度和华氏度两个属性
要求两个属性会自动进行转换，也就是说你可以给摄氏度这个属性赋值，
然后打印的华氏度是自动转换的结果。
摄氏度到华氏度：*1.8+32
我的答案：

class T:
    def __init__(self):
        self.C = 0.00
        self.H = 0.00
        
    def setC(self , value):
        self.C = value
        self.H = float('%.2f' %(self.C * 1.8 +32))

    def setH(self , value):
        self.H = value
        self.C = float('%.2f' %((self.H - 32) / 1.8)) 

    def getC(self):
        print(self.C)

    def getH(self):
        print(self.H)

    def delC(self):
        del self.C

    def delH(self):
        del self.H
        

    c = property(getC , setC , delC)

    h = property(getH , setH , delH)
'''
class Celsius:
    def __init__(self , value = 26.0):
        self.value = float(value)

    def __get__(self , instance , owner):
        return self.value 

    def __set__(self , instance , value):
        self.value = float(value)

        
class Fahrenheit:
    def __get__(self , instance , owner):
        return instance.cel * 1.8 + 32
    def __set__(self , instance , value):
        instance.cel = (float(value) - 32) / 1.8

    
class Temperature:
    cel = Celsius()
    fah = Fahrenheit()


t = Temperature()



























