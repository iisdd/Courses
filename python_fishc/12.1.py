'''
1.一个填空题
'''
list1 = ['1.Jost do It' , '2.一切皆有可能' ,
         '3.让编程改变世界' , '4.Impossible is Nothing']
list2 = ['4.阿迪达斯' , '2.李宁' , '3.鱼C工作室' , '1.耐克']

#   填空部分：
list3 = [each + i[1 : ] for i in list1 for each in list2 if i[0] == each[0]]
#   填空部分完

for each in list3:
    print(each)
