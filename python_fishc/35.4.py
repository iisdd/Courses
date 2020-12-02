import easygui as g
import os

def show_result(start_dir):
    lines = 0
    total = 0
    text = ''
    for i in source_list:
        lines = source_list[i]
        total += lines
        text += '【%s】源文件%d个，源代码%d行\n' % (i , file_list[i] , lines)
    title = '统计结果'
    msg = '宁目前共累计编写了%d行代码,完成进度：%.2f %% \n离10万行代码还差 %d行，请继续努力！' %(total,total/1000,100000-total)
    g.textbox(msg , title , text)


def calc_code(file_name):
    lines = 0
    with open(file_name) as f:
        print('正在分析文件：%s...'% file_name)
        try:
            for each_line in f:
                lines += 1
        except UnicodeDecodeError:
            pass
    return lines

def search_file(start_dir):
    os.chdir(start_dir)

    for each_file in os.listdir(os.curdir):
        ext = os.path.splitext(each_file)[1]
        if ext in target:
            lines = calc_code(each_file)
            try:
                file_list[ext] += 1  #各种类文件数
            except KeyError:
                file_list[ext] = 1

            try:
                source_list[ext] += lines  #各种类文件的行数

            except KeyError:
                source_list[ext] = lines

        if os.path.isdir(each_file):
            search_file(each_file)  #递归调用历遍文件
            os.chdir(os.pardir)  #递归完后返回上一目录

target = ['.py','.txt']
file_list = {}
source_list = {}

g.msgbox('请打开宁存放所有代码的文件夹......','统计代码量')
path = g.diropenbox('请选择宁的代码库：')

search_file(path)
show_result(path)
                

        
                
        
