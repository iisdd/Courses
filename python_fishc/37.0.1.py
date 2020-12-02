class Ticket():
    def __init__(self , weekend = False , child = False):
        self.exp = 100  #正常单价
        if weekend:
            self.inc = 1.2  #汇率
        else:
            self.inc = 1   
        if child:
            self.discount = 0.5  #折扣
        else:
            self.discount = 1
    def calcPrice(self , num):
        return self.exp * self.inc * self.discount * num

    
            
