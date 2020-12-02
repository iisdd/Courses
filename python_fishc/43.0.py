class C:
    def __init__(self , *num):
        count = 0
        for i in num:
            count += 1
        if count != 0:
            print('传入了%d个参数，分别是:' % count,end = '')
            for each in num:
                print(each , end = ' ')
        else:
            print('并没有传入参数')
