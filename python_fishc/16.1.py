def sum(*args):
    result = 0
    for each in args:
        try:
            result += each
        except TypeError:
            pass
    return result

print(sum(1 , 5 , 9 , 8 ,7 , 'ddd'))
            
