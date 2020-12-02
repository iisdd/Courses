import os

def search_file(start_dir,target):
    os.chdir(start_dir)  #改变工作目录

    for each_file in os.listdir(os.curdir):  #当前目录里的每一个文件
        if each_file == target : #是目标文件就打印地址
            print(os.getcwd() +os.sep + each_file)
        if os.path.isdir(each_file): # 是文件夹就打开继续递归
            search_file(each_file, target) #递归调用
            os.chdir(os.pardir) #递归调用后切记返回上一层目录
            
    



start_dir = input('请输入待查找的初始目录：')
target = input('请输入需要查找的目标文件：')
search_file(start_dir , target)
