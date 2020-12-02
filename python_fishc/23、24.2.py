'''
2. 还记得求回文字符串那道题吗？现在让你使用递归的方式来求解，亲还能骄傲的说我可以吗？
'''
def is_huiwen(str1):
    length = len(str1)
    if length <= 1:
        return True
    elif str1[0] == str1[- 1]:
        return is_huiwen(str1[1 : - 1])
    else:
        return False

print(is_huiwen('上海自来水来自海上'))
print(is_huiwen('双手插口袋'))
