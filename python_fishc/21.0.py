'''
0.利用filter()和lambda表达式快速求出100以内所有3的倍数
'''
print(list(filter(lambda x : x if x % 3 == 0 else None , range(1 , 101))))
