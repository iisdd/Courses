# 复杂度O(n^2)
def bubble_sort(li): # 输入无序列表,输出有序列表
    n = len(li)
    count = 0 # 趟数
    for j in range(n-1):
        for i in range(n-count-1):
            if li[i] > li[i+1]:
                li[i], li[i+1] = li[i+1], li[i] # python经典同时交换
        #     print(li,'->')
        # print('第%d次冒泡后的列表: ' % count, li)
        count += 1
    print('老版本总冒泡次数: ', count)

def bubble_sort2(li): # 增加break功能
    n = len(li)
    count = 0 # 趟数
    for j in range(n-1):
        flag = 0
        for i in range(n-count-1):
            if li[i] > li[i+1]:
                flag = 1
                li[i], li[i+1] = li[i+1], li[i] # python经典同时交换
        #     print(li,'->')
        # print('第%d次冒泡后的列表: ' % count, li)
        count += 1
        if flag == 0: # 一趟没换过就提前结束
            break
    print('新版本总冒泡次数: ', count)

bubble_sort([9,8,7,1,2,3,4,5])
bubble_sort2([9,8,7,1,2,3,4,5])


