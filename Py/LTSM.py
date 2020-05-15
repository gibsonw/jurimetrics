
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import numpy  as np
import math

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense,LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
from keras import activations

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')

plt.style.use('fivethirtyeight')

def plot_ts(*args,p_assunto= 'xx',p_rmse=.01):
    if (len(args) >= 1):
        plt.figure(figsize=(16,8))
        plt.title('Assunto: {} ; rmse = {}'.format(p_assunto,p_rmse),fontsize=12)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Quantidade',fontsize=18)
        plt.plot(args[0])
        if (len(args) == 2):
            plt.plot(args[1][['count']],'--', linewidth=2)#,valid_pred)
            plt.plot(args[1][['pred']],'-', linewidth=2)#,valid_pred)
        plt.legend(['Train','Val','Predictions'],loc='best')
        plt.show()

def get_data_jurimetrics(frequency='MS'):
    pathData =  "C:\\PUCRS\\Especialização\\Jurimetrics\\jurimetrics\\data"
    pathFileDaySubject = pathData+'\\tb_jurimetrics_adj.csv'
    #pathFileDaySubject =  "https://raw.githubusercontent.com/gibsonw/jurimetrics/master/data/count_day_subject.csv"

    date_cols = ['date']
    df_index = ['date']
    df_jurimetric_subject = pd.read_csv(pathFileDaySubject,sep=";",encoding='UTF-8', index_col=df_index,parse_dates=['date'])
    
    return (df_jurimetric_subject.groupby(['subject_decoded']).resample(frequency).sum())


def create_datasets(ds,mode='train',nPeriods=0):
    '''
    nPeriods altera o tamnho da base de teste, 
    > nPeriods menos colunas e maiores séries
    < nPeriods mais colunas e menores séries
    '''
    #create a np array do dataframe original
    dataset = ds
    dataset = dataset.astype('float32')
    #dataset = np.reshape(dataset, (-1, 1))
    train_size = math.ceil(len(dataset))
    # GW -> Never train on test data -> HOW TO NOT BE A AI IDIOT
    # np.split(dataset, [math.ceil(.85 * len(dataset))])

    X_train = []
    Y_train = []

    if (mode == 'train'):
        for i in range(nPeriods, len(dataset)):
            X_train.append(dataset[i-nPeriods:i,0])
            Y_train.append(dataset[i,0])

    if (mode == 'test'):
        for i in range(nPeriods,len(dataset)):
            X_train.append(dataset[i-nPeriods:i,0])


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    print('X_train : ',X_train.shape)
    print('Y_train : ',Y_train.shape)

    return np.array(X_train), np.array(Y_train)


def train_test_split(ds,nPeriods=12):
    #create a np array do dataframe original
    dataset_train = ds[0:len(ds)-nPeriods,:]
    dataset_test  = ds[len(ds)-nPeriods:,:]
    print('dataset:',ds.shape)
    print('dataset_train:',dataset_train.shape)
    print('dataset_test:',dataset_test.shape)
    return dataset_train,dataset_test

def lstm_timeseries_model(X_train,Y_train):

    model = Sequential()
    model.add(LSTM(100,activation='linear',return_sequences=True,input_shape=(X_train.shape[1],1)),)
    model.add(Dropout(0.05))
    model.add(LSTM(50,activation='linear',input_shape=(X_train.shape[1],1)),)
    #model.add(LSTM(50,return_sequences=False))
    model.add(Dropout(0.05))
    model.add(Dense(50,activation='linear'))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(X_train, Y_train,batch_size=1,epochs=200,verbose=0)

    print(model.summary())

    '''
    history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test), 
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)
    '''
    return model,history


#PIPELINE
# 1) Create the dataset, ensure all data is float.

dataframe = get_data_jurimetrics('MS')

l_subject_decoded = dataframe.unstack().index

subject = l_subject_decoded[5]

#%%
dataset = dataframe.loc[subject].values

nPeriods=(1*12)


# dataset inteiro
#dataset = dataframe.loc[subject].resample('W').sum()

# 2) Normalize the features.
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 4) Split into training and test sets.
# separa em base de treino e teste
#nPeriods=12
dataset_train, dataset_test = train_test_split(dataset,nPeriods)
#dataset_train = dataset
# dataset_test é usado para verificar se a predição foi boa
#dataset_test.shape
#dataset_test = scaler.inverse_transform(dataset_test)
#dataset.shape[0]==dataset_train.shape[0]+dataset_test.shape[0]

# pega a base de treino e retorna datasets pronto para testar
ds_train, ds_test = train_test_split(dataset_train,nPeriods=nPeriods)
dataset_train.shape[0]==ds_train.shape[0]+ds_test.shape[0]


# 5) Convert an array of values into a dataset matrix.
# Reshape into X=t and Y=t+1.
# Reshape input to be 3D (num_samples, num_timesteps, num_features).
# reshape input to be [samples, time steps, features]
dataset_train.shape
X_train, Y_train = create_datasets(dataset_train,mode='train',nPeriods=len(dataset_train)-nPeriods)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_train.shape
Y_train.shape

# 6) Trainning LTSM 
model,history = lstm_timeseries_model(X_train, Y_train)

# Take a slice of original train dataset, acording the shape that LTSM has trained
X_test,_X = create_datasets(dataset_train,mode='test',nPeriods=model.input_shape[1])
X_test.shape

# reshape input to be 3D 
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
X_test.shape

# run predict
y_pred = model.predict(X_test)
# convert dataset predicted in unnormalized shape 
y_pred = scaler.inverse_transform(y_pred)



# show results
train = {}
valid = {}
train[subject] = dataframe.loc[subject][:len(dataframe.loc[subject])-nPeriods]
pd_tmp = dataframe.loc[subject][-nPeriods:].astype('float32')
pd_tmp['pred'] = pd.DataFrame(y_pred[len(y_pred)-nPeriods:,:].flatten(), index=pd_tmp.index, dtype=np.float32)
valid[subject] = pd_tmp


print('Assunto:', subject)
print('Train Mean absolute error -> abse_avg:', mean_absolute_error(valid[subject]['count'], valid[subject]['pred']))
print('Train Mean squared error ->  mse:',mean_squared_error(valid[subject]['count'], valid[subject]['pred']))
print('Train Root Mean squared error ->  rmse:',np.sqrt(mean_squared_error(valid[subject]['count'], valid[subject]['pred'])))

plot_ts(train[subject].resample('MS').sum(),valid[subject].resample('MS').sum())



# %%
