import os
import easygui as g
dir_file = g.fileopenbox()
with open (dir_file) as f1:   
    file_name = os.path.basename(dir_file) 
    msg = '文件【%s】的内容如下：' % file_name
    title = '显示文件内容'
    g.textbox(msg , title , f1.read())

