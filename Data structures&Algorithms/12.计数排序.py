# 运气不好的复杂度O(n^2),运气好就是O(n)(比系统默认sort还快),空间复杂度O(n)
def count_sort(li, max_count=100):
    counter = [0 for _ in range(max_count+1)]           # 元素出现的计数器
    for val in li:
        counter[val] += 1
    li.clear()
    for idx, val in enumerate(counter):
        for i in range(val):
            li.append(idx)

import random
li = [random.randint(0, 100) for _ in range(1000)]
random.shuffle(li)
print(li)
count_sort(li)
print(li)














