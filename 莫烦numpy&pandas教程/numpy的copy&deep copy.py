import numpy as np
a = np.arange(4)
b = a
c = a
d = b
a[0] = 11
print(a,b,c,d)
d[1:3] = [22,33]
print(a,b,c,d)  # a,b,c,d都一样,指向同一内存
b = a.copy()    # deep copy ,只复制a的值,不是引用a
a[3] = 44
print(a,b,c,d)  # b没卵事
