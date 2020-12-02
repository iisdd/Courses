'''
试一下把名字用字典存下来
'''
#  名字读取
str1 = ''
with open ('p022_names.txt') as r:
    str1 += r.readline()
list1 = str1.split('","')
list1[0] = list1[0][1 : ]
list1[-1] = list1[-1][ : -1]

dict_name = {}
start = 65
for i in range(26):
    head = chr(start + i)
    temp = []
    for each in list1:
        if each[0] == head:
            temp.append(each)
    dict_name[head] = temp
    
def compare(a , b):
    len1 = len(a)
    len2 = len(b)
    for i in range (min(len1 , len2)):
        if ord(a[i]) < ord(b[i]):
            return a
        elif ord(a[i]) > ord(b[i]):
            return b
    if len1 <= len2:
        return a
    else:
        return b
def sort(a):
    temp = []
    while a:
        min1 = a[0]
        for each in a:
            min1 = compare(min1 , each)
        temp.append(min1)
        a.remove(min1)
    return temp

for each in dict_name:
    temp = dict_name[each]
    sort(temp)
    dict_name[each] = temp





    
    
