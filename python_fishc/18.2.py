'''
2. 编写一个函数 findstr()，该函数统计一个长度为 2 的子字符串在另一个字符串
中出现的次数。例如：假定输入的字符串为“You cannot improve your past,
but you can improve your future. Once time is wasted, life is wasted.”，
子字符串为“im”，函数执行后打印“子字母串在目标字符串中共出现 3 次”。
'''
def findstr():
    target = input('请输入目标字符串：')
    while 1 :
        aim = input('请输入子字符串(两个字符)：')
        if len(aim) == 2:
            break
        else:
            print('长度不对！')
    length = len(target)
    count = 0
    for each in range(length - 1):
        if (target[each] + target[each + 1]) == aim:
            count += 1
    print('出现次数：%d' % count)

findstr()
