### Python
import os
import random
import sys
import warnings
warnings.filterwarnings('ignore')


## Package
import glob 
# import keras
import IPython.display as ipd
import librosa

import librosa.display
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sns
import scipy.io.wavfile
import tensorflow as tf
import pylab
py.init_notebook_mode(connected=True)

## Keras
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, MaxPooling2D, AveragePooling2D
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.utils import to_categorical
from sklearn import metrics

from keras.models import Model, Sequential
from keras import optimizers
from keras.layers import Input, Conv1D, Conv2D,BatchNormalization, MaxPooling1D,MaxPooling2D, LSTM, Dense, Activation, Layer,Reshape


from keras.utils import to_categorical
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

## Sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

## Rest
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
# from tqdm import tqdm_notebook as tqdm

import h5py






def MFCC_feature():
    audio_duration = 7
    sampling_rate = 22050*2
    input_length = sampling_rate * audio_duration
    n_mfcc = 20
    rideo_num=1
    
    
    
    while rideo_num<6:
        
        rideo_db = pd.read_csv('./rideo_db'+str(rideo_num)+'.csv')
        rideo_db = pd.DataFrame(np.delete(rideo_db.values,0,axis=1))
         
        traindf = rideo_db.loc[rideo_db[4]=='Train']
        valdf = rideo_db.loc[rideo_db[4]=='Val']
        testdf = rideo_db.loc[rideo_db[4]=='Test']
        
        dataset_db = rideo_db.copy()
        dataset_db.columns = ['path','gender','emotion','emotion_lb','split']
        dataset_db.head()
        dataset_db.index=range(len(dataset_db.index))   
        
        data_sample= np.zeros(input_length)
        MFCC = librosa.feature.mfcc(data_sample, sr=sampling_rate, n_mfcc=n_mfcc)
        MFCC.shape
        dataset_db.split.value_counts()


        signal, sample_rate = librosa.load(dataset_db.path[0], res_type='kaiser_fast',sr=sampling_rate)
        signal,index = librosa.effects.trim(signal,top_db = 20)
        signal = scipy.signal.wiener(signal)

        if len(signal) > input_length:
            signal = signal[0:input_length]
        elif  input_length > len(signal):
            max_offset = input_length - len(signal)  
            signal = np.pad(signal, (0, max_offset), "constant")


        signal = np.array(signal).reshape(-1,1)
        
        signal.shape
        audios= np.empty(shape=(dataset_db.shape[0],20, MFCC.shape[1], 1))
        # audios= np.empty(shape=(dataset_db.shape[0],128, MFCC.shape[1], 1))

        count=0
        
        def regularit(x):
            return ((x-np.min(x))/(np.max(x)-np.min(x)))
        
        for i in range(len(dataset_db)):
            print(i)
            signal, sample_rate = librosa.load(dataset_db.path[i], res_type='kaiser_fast',sr=sampling_rate)
            signal,index = librosa.effects.trim(signal,top_db = 20)
            signal = scipy.signal.wiener(signal)
            
            if len(signal) > input_length:
                signal = signal[0:input_length]
            elif  input_length > len(signal):
                max_offset = input_length - len(signal)  
                signal = np.pad(signal, (0, max_offset), "constant")
        
            melspec = librosa.feature.mfcc(signal, sr=sample_rate,n_mfcc=20)
            logspec = melspec
            logspec_norm = regularit(logspec)*255
            logspec_norm = np.expand_dims(logspec_norm, axis=-1)
            audios[count,] = logspec_norm 
            count+=1
            



        import h5py
        with h5py.File('./Cnn_ASR_MFCC'+str(rideo_num)+'.h5', 'w') as hf:
            hf.create_dataset("Cnn_ASR_MFCC",  data=audios)
         
         
        rideo_num = rideo_num + 1
        
        
def LOGMEL_feature():
    audio_duration = 7
    sampling_rate = 22050*2
    input_length = sampling_rate * audio_duration
    n_mfcc = 20
    rideo_num=1
    
    
    
    while rideo_num<6:
        
        rideo_db = pd.read_csv('./rideo_db'+str(rideo_num)+'.csv')
        rideo_db = pd.DataFrame(np.delete(rideo_db.values,0,axis=1))
         
        traindf = rideo_db.loc[rideo_db[4]=='Train']
        valdf = rideo_db.loc[rideo_db[4]=='Val']
        testdf = rideo_db.loc[rideo_db[4]=='Test']
        
        dataset_db = rideo_db.copy()
        dataset_db.columns = ['path','gender','emotion','emotion_lb','split']
        dataset_db.head()
        dataset_db.index=range(len(dataset_db.index))   
        
        data_sample= np.zeros(input_length)
        MFCC = librosa.feature.mfcc(data_sample, sr=sampling_rate, n_mfcc=n_mfcc)
        MFCC.shape
        dataset_db.split.value_counts()



        signal, sample_rate = librosa.load(dataset_db.path[0], res_type='kaiser_fast',sr=sampling_rate)
        signal,index = librosa.effects.trim(signal,top_db = 20)
        signal = scipy.signal.wiener(signal)

        if len(signal) > input_length:
            signal = signal[0:input_length]
        elif  input_length > len(signal):
            max_offset = input_length - len(signal)  
            signal = np.pad(signal, (0, max_offset), "constant")


        signal = np.array(signal).reshape(-1,1)
        signal.shape

        # audios= np.empty(shape=(dataset_db.shape[0],20, MFCC.shape[1], 1))
        audios= np.empty(shape=(dataset_db.shape[0],128, MFCC.shape[1], 1))

        count=0
        
        def regularit(x):
            return ((x-np.min(x))/(np.max(x)-np.min(x)))
        
        for i in range(len(dataset_db)):
            print(i)
            signal, sample_rate = librosa.load(dataset_db.path[i], res_type='kaiser_fast',sr=sampling_rate)
            signal,index = librosa.effects.trim(signal,top_db = 20)
            signal = scipy.signal.wiener(signal)
            
            if len(signal) > input_length:
                signal = signal[0:input_length]
            elif  input_length > len(signal):
                max_offset = input_length - len(signal)  
                signal = np.pad(signal, (0, max_offset), "constant")
            
            melspec = librosa.feature.melspectrogram(signal, sr=sample_rate, n_mels=128,n_fft=2048,hop_length=512)   
            logspec = librosa.amplitude_to_db(melspec)
            logspec_norm = regularit(logspec)*255
            logspec_norm = np.expand_dims(logspec_norm, axis=-1)
            audios[count,] = logspec_norm 
            count+=1
                



        import h5py
        with h5py.File('./Cnn_ASR_LOGMEL'+str(rideo_num)+'.h5', 'w') as hf:
            hf.create_dataset("Cnn_ASR_LOGMEL",  data=audios)
         
         
        rideo_num = rideo_num + 1