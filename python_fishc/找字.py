import os

def print_pos(key_dict): 
    keys = key_dict.keys()  #key_dict[行数] = 出现位置
    keys = sorted(keys) # 由于字典是无序的，我们这里对行数进行排序
    for each_key in keys:
        print('关键字出现在第 %s 行，第 %s 个位置。'
              % (each_key , str(key_dict[each_key])))
    

def pos_in_line(line , key):  #哪一个位置？
    pos = []
    begin = line.find(key)
    while begin != -1:
        pos.append(begin + 1)  # 用户的角度是从1开始数的
        begin = line.find(key , begin + 1)  # 从下一个位置继续查找

    return pos
    
def search_in_file(file_name , key): #哪一行？
    f = open(file_name)
    count = 0 #记录行数
    key_dict = dict()  # 字典用来存放key所在行数与位置

    for each_line in f:
        count += 1
        if key in each_line:
            pos = pos_in_line(each_line , key)
            key_dict[count] = pos

    f.close()
    return key_dict

def search_files(key , detail):  #哪个文件
    #总函数,多个函数一起调用就像搭积木
    all_files = os.walk(os.getcwd())
    txt_files = []

    for i in all_files:
        for each_file in i[2]:
            if os.path.splitext(each_file)[1] == '.txt':
                each_file = os.path.join(i[0] , each_file)
                txt_files.append(each_file)

    for each_txt_file in txt_files:
        key_dict = search_in_file(each_txt_file , key)
        if key_dict:
            print('===============================================================')
            print('在文件【%s】中找到关键字【%s】' % (each_txt_file , key))
            if detail in ['YES' , 'Yes' , 'yes']:
                print_pos(key_dict)


key = input('请将该脚本放于待查找的文件夹内，请输入关键字：')
detail = input('请问是否需要打印关键字【%s】在文件中的具体位置（YES/NO）: ' % key)
search_files(key , detail)
                




                
    
