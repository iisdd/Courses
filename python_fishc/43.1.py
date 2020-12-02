class Word(str):
    def __init__(self , str1):
        self.str1 = str1
        self.count = 0
        for each in self.str1:
            if each == ' ':
                break
            else:
                
                self.count += 1
    def __lt__(self , other):
        return self.count < other.count

    def __le__(self , other):
        return self.count <= other.count

    def __eq__(self , other):
        return self.count == other.count

    def __ne__(self , other):
        return self.count != other.count

    def __gt__(self , other):
        return self.count > other.count

    def __ge__(self , other):
        return self.count >= other.count              

a = Word('woshi nibaba')
b = Word('dyxdashen')
