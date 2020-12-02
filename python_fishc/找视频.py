import os

def search_video(start_dir):
    list1 = ['.mp4' , '.rmvb' , '.avi','.flv']
    os.chdir(start_dir)

    for each_file in os.listdir(os.curdir):
        if os.path.splitext(each_file)[1] in list1:
            print(os.getcwd() + os.sep +each_file)
            video_list.append(os.getcwd() + os.sep +each_file+os.linesep)
           
        if os.path.isdir(each_file):
            search_video(each_file)
            os.chdir(os.pardir)
    

start_dir = input('请输入待查找的初始目录：')
program_dir = os.getcwd()
video_list = []
search_video(start_dir)
f = open(program_dir + os.sep + '视频清单.txt','w')
f.writelines(video_list)
f.close()
