try:
    f = open('my_file.txt')
    print(f.read())
except OSError as reason:
    print('出错了：' + str(reason))

finally:
    if 'f' in locals():  #局部变量里有f的话说明打开成功
        f.close()
