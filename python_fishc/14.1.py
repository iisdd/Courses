'''
 1.据说只有智商高于150的鱼油才能解开这个字符串（还原为有意义的字符串）：
 str1 = 'i2sl54ovvvb4e3bferi32s56h;$c43.sfc67o0cm99'
 '''
str1 = 'i2sl54ovvvb4e3bferi32s56h;$c43.sfc67o0cm99'
str2 = ''
for i in range(len(str1)):
	if i % 3 == 0 :
		str2 += str1[i]
print (str2)
