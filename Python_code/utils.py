import os
import numpy as np
import matplotlib.pyplot as plt

def get_fileName_newest(test_report):
    lists = os.listdir(test_report) #列出目录的下所有文件和文件夹保存到lists
    print(list)
    lists.sort(key=lambda fn:os.path.getmtime(test_report + "\\" + fn))  # 按时间排序
    file_new = os.path.join(test_report,lists[-1]) # 获取最新的文件保存到file_new
    print(file_new)

    return file_new.split('.')[0]
def delta(a, b):
    if(a == b):
        return int(1)
    else:
        return int(0)

def load_model_name(floderName,epoch = -1,modelName=None):
    """
    load a trained model in the specific filefolder
    """
    if epoch == -1:
        Folder = os.path.join(os.getcwd(),'Model_save',floderName)

        Name = get_fileName_newest(Folder)
    else:
        Name = os.path.join(os.getcwd(),'Model_save',floderName,modelName)+ str(round(epoch))
    return Name
def get_ij(n,N):
    """
        get i and j from n
    """
    if n < N+1:
        i = n+1
        j = n+1
    