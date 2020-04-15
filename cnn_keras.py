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
from sklearn.metrics import confusion_matrix,precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

## Rest
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm_notebook as tqdm

import h5py
import feature_extracting
from feature_extracting import MFCC_feature,LOGMEL_feature
import layer4_CNN
from layer4_CNN import layer4_CNN
input_duration=3


rideo_num=1
feature_kind=input('LOGMEL OR MFCC:')
if feature_kind=='LOGMEL':
    LOGMEL_feature()
elif feature_kind=='MFCC':
    MFCC_feature()
 
 

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
    
    

    import h5py
    print('./Cnn_ASR_'+str(feature_kind)+str(rideo_num)+'.h5')
    with h5py.File('./Cnn_ASR_'+str(feature_kind)+str(rideo_num)+'.h5', 'r') as hf:
      audios = hf['Cnn_ASR_'+str(feature_kind)][:]


    #----------------------------------------------------------------


    x_train = audios[(dataset_db['split'] == 'Train')]
    y_train = dataset_db.emotion_lb[(dataset_db['split'] == 'Train')]

    print(x_train.shape,y_train.shape)

    x_test = audios[(dataset_db['split'] == 'Val')]
    y_test = dataset_db.emotion_lb[(dataset_db['split'] == 'Val')]

    print(x_test.shape,y_test.shape)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))


    x_traincnn = x_train
    x_testcnn = x_test



    opt = optimizers.Adam(lr=0.001, beta_1=0.9,  beta_2=0.999, amsgrad=False)
    layer4_CNN(x_traincnn, y_train,x_testcnn, y_test,rideo_num,feature_kind,opt)




    from keras.models import model_from_json
    
    json_file = open("./Models/"+str(feature_kind)+"/CNN_ASR_MODEL"+str(rideo_num)+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)



    from keras.models import load_model

    # Returns a compiled model identical to the previous one
    loaded_model.load_weights("./Models/"+str(feature_kind)+"/CNN_ASR_MODEL"+str(rideo_num)+".h5")


    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    # x_test_data = audios[(dataset_db['split'] != 'Train')]
    # y_test_data = dataset_db.emotion_lb[(dataset_db['split'] != 'Train')]
    x_test_data = audios[(dataset_db['split'] != 'Train')]
    y_test_data = dataset_db.emotion_lb[(dataset_db['split'] != 'Train')]

    print(x_test_data.shape,y_test_data.shape)


    preds = loaded_model.predict(x_test_data,batch_size=16,verbose=1)
    preds1=preds.argmax(axis=1)
    abc = preds1.astype(int).flatten()
    predictions = (lb.inverse_transform((abc)))

    preddf = pd.DataFrame({'predictedvalues': predictions})
    preddf[:10]

    actualdf = pd.DataFrame({'actualvalues': y_test_data})
    actualdf[:10]
    actualdf.index = range(len(actualdf.index))

    finaldf = pd.concat([actualdf,preddf],axis=1)
    finaldf.head()
    
    #-----------------------------------------------------------------
      
    x_whole_data = audios
    y_whole_data = dataset_db.emotion_lb

    preds_whole = loaded_model.predict(x_whole_data,batch_size=16,verbose=1)
    preds1_whole=preds_whole.argmax(axis=1)
    fed = preds1_whole.astype(int).flatten()
    predictions_whole = (lb.inverse_transform((fed)))

    preddf_whole = pd.DataFrame({'predictedvalues': predictions_whole})
    preddf_whole[:10]

    actualdf_whole = pd.DataFrame({'actualvalues': y_whole_data})
    actualdf_whole[:10]
    actualdf_whole.index = range(len(actualdf_whole.index))
    
    finaldf_whole = pd.concat([actualdf_whole,preddf_whole],axis=1)
    finaldf_whole.head()
    

    def print_confusion_matrix(name,confusion_matrix, class_names, figsize = (9,6), fontsize=14):
        """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
        
        Arguments
        ---------
        confusion_matrix: numpy.ndarray
            The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
            Similarly constructed ndarrays can also be used.
        class_names: list
            An ordered list of class names, in the order they index the given confusion matrix.
        figsize: tuple
            A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
            the second determining the vertical size. Defaults to (10,7).
        fontsize: int
            Font size for axes labels. Defaults to 14.
            
        Returns
        -------
        matplotlib.figure.Figure
            The resulting confusion matrix figure
        """
        df_cm = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names, 
        )
        fig = plt.figure(figsize=figsize)
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
            
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('./Result/'+str(feature_kind)+'/True cs Predicted for '+str(name)+'rideo_num'+str(rideo_num)+'.png')
        plt.close('all')
        

    from sklearn.metrics import accuracy_score
    
    y_true = finaldf.actualvalues
    y_pred = finaldf.predictedvalues
    finalacc = precision_score(y_true, y_pred,average='weighted')*100
    
    
    y_true_whole = finaldf_whole.actualvalues
    y_pred_whole = finaldf_whole.predictedvalues
    finalacc_whole = precision_score(y_true_whole, y_pred_whole,average='weighted')*100
    



    from sklearn.metrics import confusion_matrix
    c = confusion_matrix(y_true, y_pred)
    c_whole = confusion_matrix(y_true_whole, y_pred_whole)


    # In[ ]:


    class_names=sorted(set(finaldf.actualvalues))
    print_confusion_matrix('test set',c, class_names)
    
    class_names_whole=sorted(set(finaldf_whole.actualvalues))
    print_confusion_matrix('whole set',c_whole, class_names_whole)
    
    
    f = open('./Result/'+str(feature_kind)+'/accuracy for rideo_db'+str(rideo_num)+'.txt','w')
    f.write('accuracy:'+str(finalacc)+'\n')
    f.write('actualdf_whole:'+str(finalacc_whole)+'\n')
    f.close()
    rideo_num=rideo_num+1
    
    
    

