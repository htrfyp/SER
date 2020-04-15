import os

path =input('searching address:')
num_dirs=0
num_files=0
num_files_rec=0

for root,dirs,files in os.walk(path):
    for each in files:
        if each[-2:] == '.o':
            print(root,dirs,each)
        num_files_rec = num_files_rec+1
    for name in dirs:
        num_dirs = num_dirs + 1
        print(os.path.join(root,name))
        
        
for fn in os.listdir(path):
    num_files  = num_files+1
    print(fn)
    
    
    
print(num_dirs)
print(num_files)
print(num_files_rec)