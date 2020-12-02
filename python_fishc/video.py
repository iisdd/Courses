import os

def search_video(start_dir):
    list1 = ['mp4' , 'rmvb' , 'avi','FLV']
    os.chdir(start_dir)
    count = 0
    for each_file in os.listdir(os.curdir):
        if os.path.splitext(each_file)[1] in list1:
            print(os.getcwd() + os.sep +each_file)
            count += 1
        if os.path.isdir(each_file):
            search_video(each_file)
            os.chdir(os.pardir)
