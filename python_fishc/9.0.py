'''
0. 设计一个验证用户密码程序，用户只有三次机会输入错误，
不过如果用户输入的内容中包含"*"则不计算在内。

'''
count = 3
while count :
    secret = 'iisddithinkcry'

    temp = input('请输入密码：')
    if temp == secret:
        print('密码正确，进入程序......')
        break
    elif '*' in temp:
        print('密码中不能含有"*"号！宁还有 %d 次机会！' % count , end = '')
    else:
        count -= 1
        print('密码输入错误！宁还有 %d 次机会！' % count , end = '')

        
