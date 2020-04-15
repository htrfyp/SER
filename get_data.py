import os
import random
import sys
import warnings
warnings.filterwarnings('ignore')
import pickle
import glob
import pandas as pd
import numpy as np
import h5py
import librosa
import librosa.display
import scipy.io.wavfile
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import keras
import keras.utils 
import np_utils
from keras.utils import to_categorical
import wave
import python_speech_features as ps
input_duration = 3


rideo_num=1
accuracy_array=np.ones([1,5])
accuracy_array_whole=np.ones([1,5])
while rideo_num<6:
    rideo_db = pd.read_csv('./rideo_db'+str(rideo_num)+'.csv')
    rideo_db = pd.DataFrame(np.delete(rideo_db.values,0,axis=1))
    

    # traindf = rideo_db.loc[rideo_db[4]='Train']
    # valdf = rideo_db.loc[rideo_db[4]='Val']
    # testdf = rideo_db.loc[rideo_db[4]='Test']
    
    traindf = rideo_db[rideo_db[4].isin(['Train'])]
    valdf = rideo_db[rideo_db[4].isin(['Val'])]
    testdf = rideo_db[rideo_db[4].isin(['Test'])]
    
    
    temptrain1=traindf[3].unique()
    temptrain2=traindf[3].value_counts()
    
    tempval1=valdf[3].unique()
    tempval2=valdf[3].value_counts()
    
    temptest1=testdf[3].unique()
    temptest2=testdf[3].value_counts()
    
    
    dataset_db = rideo_db.copy()
    dataset_db.columns = ['path','gender','emotion','emotion_lb','split']
    temp1 = dataset_db['split'].unique()
    temp2 = dataset_db['split'].value_counts()
    dataset_db.head()
    dataset_db.index = range(len(dataset_db.index))
    
    
    def regularit(x):
        return ((x-np.min(x))/(np.max(x)-np.min(x)))
        
    def read_file(filename):
        file = wave.open(filename,'r')
        params = file.getparams()
        nchannels,sampwidth,framrate,wav_length = params[:4]
        str_data = file.readframes(wav_length)
        wavedata = np.fromstring(str_data,dtype=np.short)
        time = np.arange(0,wav_length) * (1.0/framrate)
        file.close()
        return wavedata,time,framrate
        
    count = 0
    audios = np.empty(shape=(dataset_db.shape[0],300,40,3))
    for filename in tqdm(dataset_db.path):
        data,time,rate = read_file(filename)
        mel_spec = ps.logfbank(data,rate,nfilt=40)
        delta1 = ps.delta(mel_spec,2)
        delta2 = ps.delta(delta1,2)
        
        
        time = mel_spec.shape[0]
        if(time<=300):
            part = mel_spec
            delta11 = delta1
            delta21 = delta2
            part = np.pad(part,((0,300-part.shape[0]),(0,0)),'constant',constant_values=0)
            delta11 = np.pad(delta11,((0,300-delta11.shape[0]),(0,0)),'constant',constant_values=0)
            delta21 = np.pad(delta21,((0,300-delta21.shape[0]),(0,0)),'constant',constant_values=0)
            part_norm = regularit(part)
            delta1_norm = regularit(delta11)
            delta2_norm = regularit(delta21)
            audios[count,:,:,0] = part_norm
            audios[count,:,:,1] = delta1_norm
            audios[count,:,:,2] = delta2_norm
        else:
            for i in range(2):
                if(i==0):
                    begin = 0
                    end = begin+300
                else:
                    begin = time - 300
                    end = time
                    
                    
                part = mel_spec[begin:end,:]
                delta11 = delta1[begin:end,:]
                delta21 = delta2[begin:end,:]
                part_norm = regularit(part)
                delta1_norm = regularit(delta11)
                delta2_norm = regularit(delta21)
                audios[count,:,:,0] = part_norm
                audios[count,:,:,1] = delta1_norm
                audios[count,:,:,2] = delta2_norm
        
        count = count+1
        
    
    
    with h5py.File('./rideo_data_'+str(rideo_num)+'.h5','w') as hf:
        hf.create_dataset("rideo_data",  data=audios)
        
    with h5py.File('./rideo_data_'+str(rideo_num)+'.h5', 'r') as hf:
        audios = hf['rideo_data'][:]
    
        
    x_train = audios[(dataset_db['split'] == 'Train')]
    y_train = dataset_db.emotion_lb[(dataset_db['split'] == 'Train')]
    
    x_valid = audios[(dataset_db['split'] == 'Val')]
    y_valid = dataset_db.emotion_lb[(dataset_db['split'] == 'Val')]
    
    x_test = audios[(dataset_db['split'] == 'Test')]
    y_test = dataset_db.emotion_lb[(dataset_db['split'] == 'Test')]
    
    x_whole = audios
    y_whole = dataset_db.emotion_lb
        

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_whole = np.array(x_whole)
    y_whole = np.array(y_whole)
    
    
    
    lb = LabelEncoder()
    y_train = keras.utils.to_categorical(lb.fit_transform(y_train))
    y_test = keras.utils.to_categorical(lb.fit_transform(y_test))
    y_valid = keras.utils.to_categorical(lb.fit_transform(y_valid))
    y_whole = keras.utils.to_categorical(lb.fit_transform(y_whole))
    
    output = './IEMOCAP_data'+str(rideo_num)+'.pkl'
    f = open(output,'wb')
    pickle.dump((x_train,y_train,x_valid,y_valid,x_test,y_test,x_whole,y_whole),f)
    f.close()
    
    
    rideo_num = rideo_num+1
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    