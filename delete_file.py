import os
import shutil

def delete_file(top):
    for root ,dirs,files in os.walk(top,topdown=False):
        for name in files:
            os.remove(os.path.join(root,name))
        for name in dirs:
            delete_file(str(dirs))
            
            
            
if  __name__ == '__main__':
    top = input('将此路径下的文件删除：')
    delete_file(top)
