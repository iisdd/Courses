'''
    判断一串字符串中的括号是否合法
'''
def legal(s):
    stack = []
    dct = {'(':')', '[':']', '{':'}'}
    for i in s:
        if i in dct:                                    # 左括号
            stack.append(i)
        elif stack == [] or i != dct[stack.pop()]:      # 右括号
            return False
    return True

str1 = '()[]{}'
print(legal(str1))

str2 = '([{}])'
print(legal(str2))

str3 = '])}{[('
print(legal(str3))