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
from keras.layers import Input, Conv1D, Conv2D,BatchNormalization, MaxPooling1D,MaxPooling2D, LSTM, Dense, Activation, Layer,Reshape,\
    Bidirectional,GlobalAveragePooling1D,LSTM,Layer
from keras_self_attention import SeqSelfAttention

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



def layer4_CNN_bilstm(x_traincnn, y_train,x_testcnn, y_test,rideo_num,feature_kind,opt):

    # CNN I/P Config
    num_classes = len(np.unique(np.argmax(y_train, 1)))
    input_shape = x_traincnn.shape[1:]
    learning_rate = 0.0001
    decay = 1e-6
    momentum = 0.9

    #LSTM Configuration
    #num_lstm = 256


    input_shape

    model = Sequential(name='Audio_CNN_2D')

    # LFLB1
    model.add(Conv2D(filters=128, kernel_size=(2,2), strides=(1,1), padding='same', data_format='channels_last',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=128, kernel_size=(2,2), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(2,2), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    if feature_kind=='LOGMEL':
        model.add(Conv2D(filters=64, kernel_size=(2,2), strides=(1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Reshape((37,512)))
        
        model.add(Bidirectional(LSTM(units=512,return_sequences=True)))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(GlobalAveragePooling1D())
        
        model.add(Dense(units=64))
        model.add(Activation('sigmoid'))

        model.add(Dense(units=4))
        model.add(Activation('softmax'))
    elif feature_kind=='MFCC':
        model.add(Reshape((75,128)))
        
        model.add(Bidirectional(LSTM(units=128,return_sequences=True)))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(GlobalAveragePooling1D())
        
        model.add(Dense(units=64))
        model.add(Activation('sigmoid'))

        model.add(Dense(units=4))
        model.add(Activation('softmax'))
    

    # Model compilation
    # opt = optimizers.Adam(lr=0.001, beta_1=0.9,  beta_2=0.999, amsgrad=False)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])


    model.summary()







    #-----------------------------------------------------------

    #Train Config




    batch_size = 16
    num_epochs = 500





    # Model Training
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)




    earlystop = EarlyStopping(monitor='val_categorical_accuracy',patience=50,verbose=0)
    # Please change the model name accordingly.



    mcp_save = ModelCheckpoint('./Models/'+str(feature_kind)+'_BILSTM/CNN_ASR_MODEL'+str(rideo_num)+'.h5', save_best_only=True, monitor='val_categorical_accuracy')
    # mcp_save = ModelCheckpoint('./Models.h5', save_best_only=False, monitor='val_categorical_accuracy', period=5)




    cnnhistory=model.fit(x_traincnn, y_train, batch_size=batch_size, epochs=num_epochs,validation_data=(x_testcnn, y_test), callbacks=[mcp_save, lr_reduce,earlystop])




    max(cnnhistory.history['val_categorical_accuracy'])



    plt.plot(cnnhistory.history['categorical_accuracy'])
    plt.plot(cnnhistory.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./Result/'+str(feature_kind)+'_BILSTM/Accuracy for train and valid sets for rideo_db'+str(rideo_num)+'.png')
    plt.show()
    plt.close('all')
    f = open('./Result/'+str(feature_kind)+'_BILSTM/Accuracy for train and valid sets for rideo_db'+str(rideo_num)+'.txt','w')
    f.write('categorical_accuracy:'+str(cnnhistory.history['categorical_accuracy'])+'\n')
    f.write('val_categorical_accuracy:'+str(cnnhistory.history['val_categorical_accuracy'])+'\n')
    f.close()

    # Plotting the Train Valid Loss Graph

    plt.plot(cnnhistory.history['loss'])
    plt.plot(cnnhistory.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./Result/'+str(feature_kind)+'_BILSTM/Loss for train and valid sets for rideo_db'+str(rideo_num)+'.png')
    plt.show()
    plt.close('all')
    f = open('./Result/'+str(feature_kind)+'_BILSTM/Loss for train and valid sets for rideo_db'+str(rideo_num)+'.txt','w')
    f.write('loss:'+str(cnnhistory.history['loss'])+'\n')
    f.write('val_loss:'+str(cnnhistory.history['val_loss'])+'\n')
    f.close()


    # Saving the model.json

    import json

    model_json = model.to_json()
    with open("./Models/"+str(feature_kind)+"_BILSTM/CNN_ASR_MODEL"+str(rideo_num)+".json", "w") as json_file:
        json_file.write(model_json)
