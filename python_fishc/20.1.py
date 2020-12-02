'''
1. 请用已学过的知识编写程序，找出小甲鱼藏在下边这个长字符串中的密码，
密码的埋藏点符合以下规律：
    a) 每位密码为单个小写字母
    b) 每位密码的左右两边均有且只有三个大写字母
（由于我们还没有学习到文件读取方法，大家下载后拷贝过去即可）
请下载字符串文件：   string2.zip (6.17 KB, 下载次数: 21887)
'''
str2 = ''
with open ('string2.txt') as s:
    for each in s:
        str2 += each

flag1 = 0  #  前面3个大写字母
flag2 = 0  #  中间1个小写字母
flag3 = 0  #  后面3个大写字母
length = len(str2)
str1 = ''
for each in range(length):
    if str2[each].isupper():
        if flag2 == 0:
            flag1 += 1
        else:
            flag3 += 1
    else:
        if flag1 == 3:
            flag2 = 1
            target = str2[each]
            flag1 = 0
            flag3 = 0
        else:
            flag2 = 0
            flag1 = 0
            flag3 = 0
    if flag3 == 3 and str2[each + 1].islower() and target != '\n':
        print(target , end = '')
