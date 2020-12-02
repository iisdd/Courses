'''
0. 尝试利用字典的特性编写一个通讯录程序吧，功能如图：
'''
print('''|--- 欢迎进入通讯录程序 ---|
|--- 1 :查询联系人资料  ---|
|--- 2 :插入新的联系人  ---|
|--- 3 :删除已有联系人  ---|
|--- 4 :退出通讯录程序  ---|
''')
contact = {}
while 1:
    direct = input('请输入相关的指令代码：')
    if direct == '4':
        break
    elif direct == '1':
        name = input('请输入联系人姓名：')
        if name in contact:
            print(name + ' : ' + contact[name])
        else:
            print('联系人不存在！')
    elif direct == '2':
        name = input('请输入联系人姓名：')
        tele = input('请输入用户联系电话：')
        contact[name] = tele
    elif direct == '3':
        name = input('请输入删除的联系人姓名：')
        if name in contact:
            del contact[name]
        else:
            print('通讯录中不存在该联系人！')
            
        
    else:
        print('无效输入，请重新输入！')
print('|--- 感谢使用通讯录程序 ---|')
