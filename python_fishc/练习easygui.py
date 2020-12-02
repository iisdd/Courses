import easygui as g
import random
count = 2
secret = random.randint(1,10)
g.msgbox('hi,欢迎进入第一个界面的小游戏！')
guess = g.integerbox('不妨猜一下小甲鱼现在心里想的是哪个数字（1~10）：'
                         ,'数字小游戏',lowerbound = 1,upperbound = 10)
while 1:
    
    if guess != secret:
                            
        if guess > secret:
            guess = g.integerbox('大了大了,接着猜')
            count -= 1
        else:
            guess = g.integerbox('小了小了,马达马达')
            count -= 1
        if count == 0:

            if g.ccbox('次数用完了要继续玩吗？','数字小游戏'):
                count = 2
                
                guess = g.integerbox('不妨猜一下小甲鱼现在心里想的是哪个数字（1~10）：'
                         ,'数字小游戏',lowerbound = 1,upperbound = 10)
            
            else:
                break
    else:
    
        if g.ccbox('答对了！要继续玩吗？','数字小游戏'):
            guess = g.integerbox('不妨猜一下小甲鱼现在心里想的是哪个数字（1~10）：'
                         ,'数字小游戏',lowerbound = 1,upperbound = 10)
        else:
            break

    

g.msgbox('游戏结束!')
