# 把分式约分,需要用到gcd
class Fraction:
    def __init__(self, a, b):                       # a是分子,b是分母
        self.a = a
        self.b = b
        common = self.gcd(a, b)
        self.a /= common
        self.b /= common

    @staticmethod                                   # 静态方法,不用传self
    def gcd(a, b):
        while b > 0:
            a, b = b, a % b
        return a

    def __str__(self):
        return '%d/%d' % (self.a, self.b)

    def __add__(self, other):
        # Ex: 3/5 + 2/7 = 21/35 + 10/35 (通分) = 31/35 = 31/35(约分后)
        # 先通分
        tmp_a = self.a * other.b                    # 分子
        tmp_b = self.b * other.b                    # 分母
        tmp_a += other.a * self.b
        common = self.gcd(tmp_a, tmp_b)
        tmp_a /= common
        tmp_b /= common
        return Fraction(tmp_a, tmp_b)


f1 = Fraction(30, 16)
print(f1)

f2 = Fraction(5, 12)

print(f1+f2)

