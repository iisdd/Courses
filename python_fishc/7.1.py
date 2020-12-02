'''
1. Python 的作者在很长一段时间不肯加入三元操作符就是怕跟 C 语言一样搞出国际乱码大赛，
蛋疼的复杂度让初学者望而生畏，不过，如果你一旦搞清楚了三元操作符的使用技巧，
或许一些比较复杂的问题反而迎刃而解。

请将以下代码修改为三元操作符实现：
x, y, z = 6, 5, 4
if x < y:
    small = x
    if z < small:
        small = z
elif y < z:
    small = y
else:
    small = z
复制代码
'''
x , y , z = 6 , 5 , 4
small = (x if x < y and x < z else (y if y < z else z) )
print(small)
