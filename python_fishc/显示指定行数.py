def content(address , start , end):
    f1 = open(address)
    if start == '开始':
        start = 0
    else:
        start = int(start)
    if end == '末尾':
        end = -1
    else:
        end = int(end)
    if end > 0:
        for each in range(0,end):
            if each >= start - 1:
                print(f1.readline(),end='')
            else:
                f1.readline()
    else:
        for each in range(0,start+1):
            f1.readline()
        print(f1.read())
        
    f1.close()
    

address = input(r'''请输入要打开的文件(C:\\test.txt):''')
num = input('请输入需要显示的行数【格式如 13：21 或：21 或 21：】：')
(start , end) = num.split(':',1)
if start == ' ':
    start = '开始'
else:
    start = str(start)
if end == ' ':
    end = '末尾'
else:
    end = str(end)

print('文件'+address+'从第'+start+'行到第'+end+'行的内容如下：' )
content(address , start , end)
