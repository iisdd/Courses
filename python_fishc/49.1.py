'''
1. 10 以内的素数之和是：2 + 3 + 5 + 7 = 17，那么请编写程序，
计算 2000000 以内的素数之和？
'''
import math 
def is_crime(num):
    cs = int(math.sqrt(num) + 1)
    while num % cs:
        
        if cs == 2:
            return True
        cs -= 1
    return False

def find_crimes():
    
    target_num = 3
    while 1 :
        if is_crime(target_num):
            yield target_num
            target_num += 2
        else:
            target_num += 2
        if target_num > 2000000:
            break
count = 2
for i in find_crimes():
    count += i
print(count)
            
            
