import heapq    # q: queue优先队列
import random

li = list(range(20))
random.shuffle(li)

print('原列表: ', li)
heapq.heapify(li)                           # 建堆,小根堆
print('建堆后: ', li)

n = len(li)
for _ in range(n):
    print(heapq.heappop(li), end=',')       # 每次弹出最小值,如果要排序就把pop的值存入新列表


