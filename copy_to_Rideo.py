import os
import shutil

def copy(top):
    for root ,dirs,files in os.walk(top,topdown=False):
        for name in files:
            shutil.copy(os.path.join(root,name), "./Rideo")
        for name in dirs:
            copy(str(dirs))
            
            
            
if  __name__ == '__main__':
    #top = input('将此路径下的文件复制到Rideo：')
    i = 1
    while i < 6:
        top = './IEMOCAP/Session'+str(i)+'/sentences/wav'
        copy(top)
        i = i + 1