def hanoi(n , x , y , z ):
    count = 0

    if n == 1 :
        print(x,'-->',z)
        count += 1

    else:
       
        #   将前n - 1个盘子从x移动到y上
        count += hanoi(n - 1 , x , z , y)
        print(x,'-->',z) #  将最底下的最后一个盘子从x移动到z上
        count += 1

        
        #   将y上的n - 1 个盘子移动到z上
        count += hanoi(n - 1 , y , x , z)

    return count
        
    
n = int(input('请输入汉诺塔的层数:'))

print('移动次数：' , hanoi(n , 'x' , 'y' , 'z' ))

list1 = [3 , 7 , 15 , 31 , 63 , 127]
