class Ticket:
    adult = 0
    kid = 0
    day = ''
    def setnum(self):
        self.adult = int(input('请输入成人数:'))
        self.kid = int(input('请输入小孩数:'))

    def sum(self):
        self.day = input('是否为周末yes/no? \n')

        if self.day == 'yes':
            return (self.kid * 0.5 + self.adult ) * 120
        else:
            return (self.kid * 0.5 + self.adult ) * 100
        
    
