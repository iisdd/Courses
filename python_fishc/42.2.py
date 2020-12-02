class Nstr(int):
    def __new__(cls , string):
        if isinstance(string , str):
            
            count = 0
            for each in string:
                count += ord(each)
            string = count
        return int.__new__(cls , string)
    
