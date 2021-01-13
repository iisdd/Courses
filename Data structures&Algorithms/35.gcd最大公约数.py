# 辗转相除法,到一方余数为0停止
def gcd(a, b):
    while b > 0:
        a, b = b, a%b
    return a
print(gcd(12, 16))


