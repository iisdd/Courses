'''
1. 关于最后提到的长字符串（三重引号字符串）
其实在 Python3 还可以这么写，不妨试试，然后比较下哪种更方便？
>>> string = (
"我爱鱼C，\n"
"正如我爱小甲鱼，\n"
"他那呱唧呱唧的声音，\n"
"总缠绕于我的脑海，\n"
"久久不肯散去……\n")
'''
str1 = '''又是一个深夜凌晨我不想睡
多久才能够买房还要赚多少出场费
玩说唱怎么养活自己也许你说的对
可谁想当个没有出息的窝囊废
'''
str2 = (
    '我想过很多种的死法但我现在依然活着\n'
    '没人相信我能做音乐我自己就是伯乐\n'
    '不想朝九晚五你就说我顽固走弯路\n'
    '你只在乎我的收入不会在乎我的专注\n')
print (str1 , end = '')
print (str2)