class Nint(int):
    def __new__(cls , arg = 0):
        if isinstance(arg , str):   #如果是字符串就算ACII码值
            count = 0
            for each in arg:
                count += ord(each)
            arg = count
        return int.__new__(cls , arg)   #如果是数字就直接继承int
