'''
0. 完善第二个改进要求（为用户提供三次机会尝试，机会用完或者用户猜中答案均退出循环）
并改进视频中小甲鱼的代码。(猜数字小游戏)
'''
import random 
def guess():
    count = 3
    secret = random.randint(1 , 10)
    while count :
        try:
            temp = int(input('给爷猜(1-10)：'))
                    
            if temp == secret:
                print('强！')
                break
            elif temp < secret:
                print('小咯！')
            else:
                print('大咯！')
            count -= 1
            if count == 0:
                print('答案是:',secret,'菜！')
        except ValueError:
            print('给爷爬！')
        

guess()
