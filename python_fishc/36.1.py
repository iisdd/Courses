class Rectangle:
    chang = 5.00
    kuan = 4.00

    def getRect(self):
        print('这个矩形的长是：%.2f,宽是%.2f' % (self.chang , self.kuan))

    def setRect(self):
        print('请输入矩形的长和宽...')
        self.chang = float (input('长：'))
        self.kuan = float(input('宽：'))

    def getArea(self):
        print(self.chang * self.kuan)
