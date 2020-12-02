'''编一个华容道游戏嗷'''
import random
import easygui as g
class Game:
        def __init__(self):
                self.list1 =[ 1 , 2 , 3 , 4 ,
                              5 , 6 , 7 , 8 ,
                              9 , 10 , 11 , 12 ,
                              13 , 14 , 15 , ' ']
                self.pos = 15
        def getpos(self):
                return self.list1.index(' ')

        def __repr__(self):
                return str(self.list1)
        def move(self , cmd):
                if (cmd == 'up') and (self.pos < 12):
                        self.list1[self.pos] = self.list1[self.pos + 4]
                        self.list1[self.pos + 4] = ' '

                elif (cmd == 'down') and (self.pos > 3):
                        self.list1[self.pos] = self.list1[self.pos - 4]
                        self.list1[self.pos - 4] = ' '

                elif (cmd == 'left') and (self.pos % 4 != 3):
                        self.list1[self.pos] = self.list1[self.pos + 1]
                        self.list1[self.pos + 1] = ' '

                elif (cmd == 'right') and (self.pos % 4):
                        self.list1[self.pos] = self.list1[self.pos - 1]
                        self.list1[self.pos - 1] = ' '
                self.pos = self.getpos()
                
                        
c = Game()
#   界面部分
#   初始化拼图，随机走500步保证能解出来
for i in range(500):
        choice = random.choice(['up' , 'down' , 'left' , 'right'])
        c.move(choice)
#界面4乘4,这就是我的对齐路线哒！！！
def getstr():
        str1 = ''
        str2 = ''
        str3 = ''
        str4 = ''
        for i in range(0 , 4):
                if c.list1[i] == ' ':
                        tail = '  '
                elif int(c.list1[i]) < 10:
                        tail = '  '
                else :
                        tail = ' '
                str1 += str(c.list1[i]) + tail
        for i in range(4 , 8):
                if c.list1[i] == ' ':
                        tail = '  '
                elif int(c.list1[i]) < 10:
                        tail = '  '
                else :
                        tail = ' '
                str2 += str(c.list1[i]) + tail
        for i in range(8 , 12):
                if c.list1[i] == ' ':
                        tail = '  '
                elif int(c.list1[i]) < 10:
                        tail = '  '
                else :
                        tail = ' '
                str3 += str(c.list1[i]) + tail
        for i in range(12 , 16):
                if c.list1[i] == ' ':
                        tail = '  '
                elif int(c.list1[i]) < 10:
                        tail = '  '
                else :
                        tail = ' '
                str4 += str(c.list1[i]) + tail

        return str1 + '\n' + str2 + '\n' + str3 + '\n' + str4

def start():
        while c.list1 != [ 1 , 2 , 3 , 4 ,5 , 6 , 7 , 8 ,9 , 10 ,
                           11 , 12 ,13 , 14 , 15 , ' ']:
                str1 = getstr()
                choice = g.choicebox(msg = str1,title = '华容道' ,
                            choices = ['up' , 'down' , 'left' , 'right'])
                if not choice:
                        break
                c.move(choice)
        try:
                nextgame = g.choicebox('通关了牛逼嗷！是否继续游戏？' , chocies = ['继续' , '退出'])
                if nextgame == '继续':
                        start()
        except TypeError:
                g.msgbox('拜了个拜！' , '退出游戏' , ok_button = '爬')
start()


