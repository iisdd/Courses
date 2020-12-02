import os
import easygui as g
dir_file = g.fileopenbox()
with open (dir_file) as f1:   
    file_name = os.path.basename(dir_file) 
    msg = '文件【%s】的内容如下：' % file_name
    title = '显示文件内容'
    str1 = g.textbox(msg , title , f1.read())
with open (dir_file,'r') as f1:
    old = f1.read()
    if str1 != old:
        msg = '检测到文件内容发生改变，请选择以下操作：'
        title = '警告'
        choice = ('覆盖保存','放弃保存','另存为...')
        save = g.buttonbox(msg , title , choice)
        if save == '另存为...':
            file_new = g.filesavebox()
            with open(file_new,'w') as f3:
                f3.writelines(str1)

        with open(dir_file,'w') as f2:


            if save == '覆盖保存':
                f2.writelines(str1)
            else:
                f2.writelines(old)
        

            
