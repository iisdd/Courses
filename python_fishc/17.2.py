def shier(n):
    list1 = []
    while n :
        list1.append(n % 2)
        n //= 2
    str1 = '0b'
    length = len(list1)
    for i in range(length):
        str1 += str(list1.pop())
    return str1
