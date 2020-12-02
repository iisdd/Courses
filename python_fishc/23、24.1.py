'''
1. 写一个函数get_digits(n)，将参数n分解出每个位的数字并按顺序存放到列表中。
举例：get_digits(12345) ==> [1, 2, 3, 4, 5]
'''
list1 =[]
def get_digits(n):
    if n:
        list1.append(n % 10)
        return get_digits(n // 10)

get_digits(54321)
list1.sort()
print(list1)
