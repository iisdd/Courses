'''
2. 尝试写代码实现以下截图功能
>>>
请输入一个整数：8
       ********
      *******
     ******
    *****
   ****
  ***
 **
*
>>>
'''
def star(time):
    while time:
        print(' '*(time-1) + '*'*time)
        time -= 1
while 1:
    try:
        temp = int(input('请输入一个整数：'))
        star(temp)
        break
    except ValueError:
        print('整数嗷！')

