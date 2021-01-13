# 贪心算法经典案例: 活动选择问题
# 在一个场地举行活动,每个可选的活动都有起始时间Si和结束时间Fi,同一时间场地只能举行一个活动,
# 问怎么安排活动才能使场地举行最多的活动
# 贪心结论: 最先结束的活动一定是安排方案中的

activities = [(1,4), (5,7), (8,12), (2,14), (4,6), (3,5), (0,6), (3,9), (12,16), (6,10), (5,9), (8,11)]
activities.sort(key=lambda x:x[1])
print(activities)
def act_choose(a):
    res = [a[0]]
    for start, end in a:
        if start >= res[-1][1]:
            res.append((start, end))
            cur_time = end
    return res

print(act_choose(activities))
