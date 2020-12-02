class Nstr(str):
    def __sub__(self , other):
        temp = self.replace(other , '')
        return temp
        
    
a = Nstr('wdnmd nmsl sb')
b = Nstr('m')
print(a - b)
