class Stack():
    
    def __init__(self):
        self.dict1 = {0:'C' , 1:'h' , 2:'s' , 3:'i' , 4:'F'}
    def isEmpty(self):
        if len(self.dict1) == 0:
            return True
        else:
            return False

    def push(self , str1):
        top = len(self.dict1) 
        self.dict1[top] = str1 

    def pop(self):
        if not len(self.dict1):
            print('栈为空！')
        else:
            print(self.dict1.pop(len(self.dict1)-1))

    def top(self):
        if not len(self.dict1):
            print('栈为空！')
        else:
            print(self.dict1[len(self.dict1 ) - 1])

    def bottom(self):
        if not len(self.dict1):
            print('栈为空！')
        else:
            print(self.dict1[0])

    
        
