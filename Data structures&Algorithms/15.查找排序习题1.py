'''
    给定两个字符串 s 和 t 判断t是否为s重新排列后组成的单词
'''
def same(s, t):
    dct_s = {}
    for i in s:
        dct_s[i] = dct_s.get(i, 0) + 1
    for j in t:
        if j not in dct_s:
            return False
        dct_s[j] -= 1
    # 有可能两个空字符串: "",字典的值就是空集合
    return set(dct_s.values()) == {0} or set(dct_s.values()) == set()
s1 = 'anagram'
t1 = 'nagaram'
print(same(s1, t1))

s2 = 'car'
t2 = 'rat'
print(same(s2, t2))

s3 = ''
t3 = ''
print(same(s3, t3))