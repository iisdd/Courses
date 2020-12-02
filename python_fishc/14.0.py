'''0. 请写一个密码安全性检查的代码代码：check.py'''
# 密码安全性检查代码
#
# 低级密码要求：
#   1. 密码由单纯的数字或字母组成
#   2. 密码长度小于等于8位
#
# 中级密码要求：
#   1. 密码必须由数字、字母或特殊字符（仅限：~!@#$%^&*()_=-/,.?<>;:[]{}|\）任意两种组合
#   2. 密码长度不能低于8位
#
# 高级密码要求：
#   1. 密码必须由数字、字母及特殊字符（仅限：~!@#$%^&*()_=-/,.?<>;:[]{}|\）三种组合
#   2. 密码只能由字母开头
#   3. 密码长度不能低于16位

def check():
    symbol = '~!@#$%^&*()_=-/,.?<>;:[]{}|\\'
    test = input('请输入需要检查的密码组合：')
    length = len(test)
    flag = 0
    notice = '''请按以下方式提升宁的密码安全级别：
    1.密码必须由数字、字母及特殊字符三种组合
    2.密码只能由字母开头
    3.密码长度不能低于16位'''
    print('宁的密码安全级别评定为：' , end ='')

    for each in test:
        if each in symbol:
            flag = 1
            break
        
    if test.isalnum() or length <= 8:
        print('低')
        print(notice)

    
    elif test[0].isalpha() and length >= 16 and flag == 1 :
        print('高')
        print('请继续保持')
        return True

    else:
        print('中')
        print(notice)

while 1 :
    if check():
        break
    
