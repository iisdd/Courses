'''
2. 上一题打印的样式不是很好，
能不能修改一下代码打印成下图的样式呢？【请至少使用两种方法实现】
小甲鱼 88
黑夜 90
迷途 85
怡静 90
秋舞斜阳 88
'''
#   方法一：
member = ['小甲鱼', 88, '黑夜', 90, '迷途', 85, '怡静', 90, '秋舞斜阳', 88]
length = len(member)
for i in range(0 , length , 2):
    print(member[i] , end = ' ')
    print(member[i + 1])

#   方法二：
member = ['小甲鱼', 88, '黑夜', 90, '迷途', 85, '怡静', 90, '秋舞斜阳', 88]
length = len(member)
for i in range(length):
    if i % 2 == 0:
        print(member[i] , member[i + 1])
