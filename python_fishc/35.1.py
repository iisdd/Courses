import easygui as g

msg = '请填写以下联系方式'
title = '账号中心'
fieldnames = ['*用户名' , '*真实姓名' , '固定电话' , '*手机号码' , 'QQ'
              , '*E-mail'] # 域名
fieldvalues = []  # 域值
fieldvalues = g.multenterbox(msg , title , fieldnames)


while 1:
    if fieldvalues == None:   #用户取消操作返回None
        break
    errmsg = ''
    for i in range(len(fieldnames)):
        option = fieldnames[i].strip()   #去除头尾空格的fieldnames
        if fieldvalues[i].strip() == '' and option[0] == '*':   # 必须填又没填
            errmsg += ('【%s】为必填项。\n\n' % fieldnames[i])

    if errmsg == '':
        break
    fieldvalues = g.multenterbox(errmsg , title , fieldnames , fieldvalues)

print('用户资料如下： %s' %str(fieldvalues))

            
        
