class Nint(int):
    def __new__(cls , temp):
        if type(temp) == int or type(temp) == float :
            return int.__new__(cls , temp)
        else:
            count = 0
            for each in temp:
                count += ord(each)
            temp = count
            return int.__new__(cls , temp)
        
