import random
secret=random.randint(1,9)
""" 用python设计第一个游戏 """
print("不妨猜一下小甲鱼现在心里想的是哪个数字，")

    
guess = 0
a=0

while (guess!=secret) and (a<3):   #这里的guess相当于Q0
    temp = input()
    
    if temp.isdigit():
            
            guess =int(temp)
            if guess == secret:
                print("你是小甲鱼心里的蛔虫吗？！")
                print("哼，猜中了也没奖励！")
            else:
                
        

                if guess > secret:
                    print("大了")
                else:
                    print("小了")
                if a<2:
                    print('再试一次吧')
                    
          
                else:
                    print('机会用完咯')
                    
    else:
        print('请输入一个整数：')
    a=a+1
print("游戏结束，不玩啦^-^")

