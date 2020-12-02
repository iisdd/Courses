'''
0. 猜想一下 min() 这个BIF的实现过程
'''
def minx(*arg):
    xiuxiu = arg[-1]
    for each in arg:
        if each < xiuxiu:
            xiuxiu = each
    return xiuxiu
print(minx(1 , 5 , 9 , 8 ,7))
