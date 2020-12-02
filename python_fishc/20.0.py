'''
0. 请用已学过的知识编写程序，统计下边这个长字符串中各个字符出现的次数
并找到小甲鱼送给大家的一句话。
（由于我们还没有学习到文件读取方法，大家下载后拷贝过去即可）
请下载字符串文件：   string1.zip (55.49 KB, 下载次数: 23394)
'''
str1 = ''
with open ('string1.txt') as s:
    for each in s:
        str1 += each
for each in str1:
    if each.isalpha():
        print(each , end ='')
        
