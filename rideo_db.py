import os
import random
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from tqdm import tqdm
import h5py

input_duration = 3

labeldata = np.loadtxt('./label.txt',dtype=str,encoding='utf8')
xxxdata = np.loadtxt('./xxxdata.txt',dtype=str,encoding='utf8')
rideo_path = "./Rideo"
rideo_list=os.listdir(rideo_path)
rideo_list.sort()
k=0

# count = 0
# for j in rideo_list:
    # k=j.split('.')[0]
    # if k in xxxdata:
        # count = count+1
        # os.remove(str("./Rideo")+"/"+str(j))
        # print("remove"+str(j))
        # print('total deleta:'+str(count))         

# for j in rideo_list:
    # k=j.split('.')[0]
    # if k  in labeldata[:,0]:
        # os.remove(str("./Rideo")+"/"+str(j))
        # print(str(j))
  
#a=input('输入以继续') 
 
rideo_db = pd.DataFrame(columns=['path','gender','emotion','emotion_lb'])
count = 0

w = 0
rideo_list.sort()
labeldata.sort()
for i in rideo_list:
    if w > len(rideo_list):
        break
    k = i.split('.')[0]
    if k == str(labeldata[w][0]):
        print("yes"+" "+k+" "+str(labeldata[w][0]))
        
        emtion_num = -10
        if labeldata[w][1] == 'neu':
            emtion_num=1
        elif labeldata[w][1] == 'hap':
            emtion_num=2
        elif labeldata[w][1] == 'sad':
            emtion_num=3
        elif labeldata[w][1] == 'ang':
            emtion_num=4
        elif labeldata[w][1] =='sur':
            emtion_num=5
        elif labeldata[w][1] == 'fea':
            emtion_num=6
        elif labeldata[w][1] == 'dis':
            emtion_num=7
        elif labeldata[w][1] == 'fru':
            emtion_num=8
        elif labeldata[w][1] == 'exc':
            emtion_num=9
            
        gender_num = -10
        genderpart = labeldata[w][0]
        gnm = genderpart.split('_')[0]
        if 'F' in gnm:
            gender_num = 1
        else:
            gender_num = 2
    
    else:
        print("no"+" "+k+" "+str(labeldata[w][0]))
        sys.exit(0)
        
        
        
    rideo_db.loc[count] = [str(rideo_path) + "/" + str(i),gender_num, emtion_num,labeldata[w][1]]
    count = count+1
    
    w = w+1
    
    
    
rideo_db = rideo_db.values
rideo_db = pd.DataFrame(rideo_db)
print(rideo_db)
# rideo_db.drop(rideo_db.drop.index[rideo_db.drop[3]=='sur'],inplace=True)
# rideo_db.drop(rideo_db.drop.index[rideo_db.drop[3]=='fea'],inplace=True)
# rideo_db.drop(rideo_db.drop.index[rideo_db.drop[3]=='oth'],inplace=True)
# rideo_db.drop(rideo_db.drop.index[rideo_db.drop[3]=='dis'],inplace=True)
# rideo_db.drop(rideo_db.drop.index[rideo_db.drop[3]=='fru'],inplace=True)
# rideo_db.loc[rideo_db[3]=='hap',[2,3]]=9,'exc'



rideo_db=rideo_db[~rideo_db[3].isin(['sur'])]
rideo_db=rideo_db[~rideo_db[3].isin(['fea'])]
rideo_db=rideo_db[~rideo_db[3].isin(['oth'])]
rideo_db=rideo_db[~rideo_db[3].isin(['dis'])]
rideo_db=rideo_db[~rideo_db[3].isin(['fru'])]
rideo_db.loc[rideo_db[3]=='hap',[2,3]]=9,'exc'





rideo_db.head()


random.seed(337)

rideo_db['split'] = 'Unknown'
rideo_db1 = pd.DataFrame(data=rideo_db)
rideo_db2 = pd.DataFrame(data=rideo_db)
rideo_db3 = pd.DataFrame(data=rideo_db)
rideo_db4 = pd.DataFrame(data=rideo_db)
rideo_db5 = pd.DataFrame(data=rideo_db)       
rideo_db_array = rideo_db.values


xx=0
for x in range(rideo_db.shape[0]):

    compare = random.random()
    #compare = float(compare)
    
    if compare > 0.8:
        (rideo_db1.values)[xx,4] = 'Test'
    elif compare > 0.6:
        (rideo_db2.values)[xx,4] = 'Test'
    elif compare > 0.4:
        (rideo_db3.values)[xx,4] = 'Test'
    elif compare > 0.2:
        (rideo_db4.values)[xx,4] = 'Test'
    else :
        (rideo_db5.values)[xx,4] = 'Test'

    xx = xx + 1
    
random.seed(3372)

val_number=0.7

rideo_db1 = rideo_db1.values
for index in range(rideo_db1.shape[0]):
    compare = random.random()
    if rideo_db1[index,4] != 'Test':
       rideo_db1[index,4] = 'Train'
    else:
       if compare > val_number:
          rideo_db1[index,4] = 'Val'
       else:
          rideo_db1[index,4] = 'Test'
rideo_db1 = pd.DataFrame(rideo_db1)
pd.DataFrame(rideo_db1).to_csv(path_or_buf='./rideo_db1.csv')


rideo_db2 = rideo_db2.values
for index in range(rideo_db2.shape[0]):
    compare = random.random()
    if rideo_db2[index,4] != 'Test':
       rideo_db2[index,4] = 'Train'
    else:
       if compare > val_number:
          rideo_db2[index,4] = 'Val'
       else:
          rideo_db2[index,4] = 'Test'
rideo_db2 = pd.DataFrame(rideo_db2)
pd.DataFrame(rideo_db2).to_csv(path_or_buf='./rideo_db2.csv')


rideo_db3 = rideo_db3.values
for index in range(rideo_db3.shape[0]):
    compare = random.random()
    if rideo_db3[index,4] != 'Test':
       rideo_db3[index,4] = 'Train'
    else:
       if compare > val_number:
          rideo_db3[index,4] = 'Val'
       else:
          rideo_db3[index,4] = 'Test'
rideo_db3 = pd.DataFrame(rideo_db3)
pd.DataFrame(rideo_db3).to_csv(path_or_buf='./rideo_db3.csv')


rideo_db4 = rideo_db4.values
for index in range(rideo_db4.shape[0]):
    compare = random.random()
    if rideo_db4[index,4] != 'Test':
       rideo_db4[index,4] = 'Train'
    else:
       if compare > val_number:
          rideo_db4[index,4] = 'Val'
       else:
          rideo_db4[index,4] = 'Test'
rideo_db4 = pd.DataFrame(rideo_db4)
pd.DataFrame(rideo_db4).to_csv(path_or_buf='./rideo_db4.csv')


rideo_db5 = rideo_db5.values
for index in range(rideo_db5.shape[0]):
    compare = random.random()
    if rideo_db5[index,4] != 'Test':
       rideo_db5[index,4] = 'Train'
    else:
       if compare > val_number:
          rideo_db5[index,4] = 'Val'
       else:
          rideo_db5[index,4] = 'Test'
rideo_db5 = pd.DataFrame(rideo_db5)
pd.DataFrame(rideo_db5).to_csv(path_or_buf='./rideo_db5.csv')


















        
        
        
        
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            