'''0. 课堂上小甲鱼说可以利用分片完成列表的拷贝 list2 = list1[:]，
那事实上可不可以直接写成 list2 = list1 更加简洁呢？
'''
#   举个例子：
list1 = [1 , 9 , 5 , 7 , 6 , 2]
list2 = list1[:]
list3 = list1
list1.sort()
print('母体列表1：' + str(list1))
print('copy列表2：' + str(list2))
print('墙头草列表3：' + str(list3))
