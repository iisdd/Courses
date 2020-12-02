'''
1. 编写一个函数，分别统计出传入字符串参数（可能不只一个参数）的英文字母、
空格、数字和其它字符的个数。

'''
def statistic(test):
    type_name = {'英文字母':0 , '数字':0 , '空格':0 , '其他字符':0}
    for each in test:
        if each.isalpha():
            type_name['英文字母'] += 1
        elif each.isdigit():
            type_name['数字'] += 1
        elif each.isspace():
            type_name['空格'] += 1
        else:
            type_name['其他字符'] += 1
    for each in type_name:
        print(each+'%s'%type_name[each]+'个,' , end = '')
def count(*args):
    seq = 1
    for each in args:
        if seq == 1:
            print('第 %d 个字符串共有：' % seq, end = '')
        else:
            print('\n第 %d 个字符串共有：' % seq, end = '')
        statistic(each)
        seq += 1




count('I love fishc.com.' , 'I love you, you love me.')
