
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
from keras.callbacks import EarlyStopping

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
            plt.plot(args[1][['count']],'--', linewidth=1)#,valid_pred)
            plt.plot(args[1][['pred']],'-', linewidth=1)#,valid_pred)
        plt.legend(['Train','Val','Predictions'],loc='best')
        plt.show()

'''
def run_LTSM(*args):
    _perc_train = .85
    #create a np array do dataframe original
    ds = args[1].loc[args[0]].values
    print('shape dateset do assunto :',ds.shape)
    training_data_len = math.ceil(len(ds) * _perc_train)

    #Scacle the data

    scaler = MinMaxScaler(feature_range=(0,1))
    sc_ds = scaler.fit_transform(ds)

    train_data = sc_ds[0:training_data_len,:]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
    
    #print('x_train',x_train)
    #print('y_train',y_train)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print('shape x_train :',x_train.shape)
    print('shape y_train :',y_train.shape)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    print('shape x_train after reshape :',x_train.shape)

    # build LTSM model
    model = Sequential()

    model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50,return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(1))

    #compile
    model.compile(optimizer='adam',loss='mean_squared_error')

    #train
    model.fit(x_train,y_train,batch_size=1,epochs=1)

    # create the testing data set
    # create a new array containing scaled values from index 
    print('shape do teste :',sc_ds.shape)
    test_data = sc_ds[training_data_len-60:,:]
    print('shape do test com os ultimos 60 registros :',test_data.shape)
    #create data sets x_test and y_test

    x_test = []
    y_test = ds[training_data_len:,:]
    print('shape do dataset y :',sc_ds.shape)
    y_test.shape

    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])

    x_test = np.array(x_test)

    print('matrix com lag de -1 no vertice z :',x_test.shape)

    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    y_pred = model.predict(x_test)
    y_pred = scaler.inverse_transform(y_pred)

    # get de RMSE

    rmse = np.sqrt(np.mean(y_pred - y_test)**2)
    rmse

    train = args[1].loc[args[0]][:training_data_len]
    valid = pd.DataFrame(args[1].loc[args[0]][training_data_len:],dtype=np.float32)
    valid['pred'] = pd.DataFrame(y_pred.flatten(), index=valid.index, dtype=np.float32)

    print('Registros Train {}, Registros Pred {}, Registros DF Original {}'.format(df_train.shape[0],df_valid.shape[0],ds.shape[0]))

    return (rmse,train,valid)
'''


#Create the dataset, ensure all data is float.
#Normalize the features.
#Split into training and test sets.
#Convert an array of values into a dataset matrix.
#Reshape into X=t and Y=t+1.
#Reshape input to be 3D (num_samples, num_timesteps, num_features).


def get_data_jurimetrics(frequency='MS'):
    pathData =  "C:\\PUCRS\\Especialização\\Jurimetrics\\jurimetrics\\data"
    pathFileDaySubject = pathData+'\\tb_jurimetrics_adj.csv'
    #pathFileDaySubject =  "https://raw.githubusercontent.com/gibsonw/jurimetrics/master/data/count_day_subject.csv"

    date_cols = ['date']
    df_index = ['date']
    df_jurimetric_subject = pd.read_csv(pathFileDaySubject,sep=";",encoding='UTF-8', index_col=df_index,parse_dates=['date'])
    
    return (df_jurimetric_subject.groupby(['subject_decoded']).resample(frequency).sum())


def create_datasets(ds,mode=train,nPeriods=0):
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
        if (math.ceil(train_size/2)-nPeriods < 2):
            nColumns = math.ceil(train_size/2)
            print('nPeriods muito grande, parametro será atualizado com valor : {}'.format(nColumns))
        else:
            nColumns = (math.ceil(train_size/2)-nPeriods)

        for i in range(nColumns, train_size):
            X_train.append(dataset[i-nColumns:i,0])
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
    model.add(LSTM(50, input_shape=(X_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    history = model.fit(X_train, Y_train,batch_size=1,epochs=300)
    
    '''
    history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test), 
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)
    '''
    return model,history



dataframe = get_data_jurimetrics('MS')

l_subject_decoded = dataframe.unstack().index
subject = l_subject_decoded[2]
# dataset inteiro
dataset = dataframe.loc[subject].values

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# separa em base de treino e teste
nPeriods=12
dataset_train, dataset_test = train_test_split(dataset,nPeriods)


# dataset_test é usado somente para verificar se a predição foi boa
dataset_test.shape
dataset_test = scaler.inverse_transform(dataset_test)
dataset.shape[0]==dataset_train.shape[0]+dataset_test.shape[0]

# pega a base de treino e retorna datasets pronto para testar
ds_train, ds_test = train_test_split(dataset_train,nPeriods=12)

dataset_train.shape[0]==ds_train.shape[0]+ds_test.shape[0]


X_train, Y_train = create_datasets(ds_train,mode='train',nPeriods=15)

# reshape input to be [samples, time steps, fe
# atures]
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_train.shape

model,history = lstm_timeseries_model(X_train, Y_train)



X_test,_X = create_datasets(dataset_train,mode='test',nPeriods=model.input_shape[1])

# vou passar a base inteira de treino dataset_train
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
X_test.shape


y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)



train = dataframe.loc[subject][:len(dataframe.loc[subject])-nPeriods]
valid = dataframe.loc[subject][-nPeriods:].astype('float32')
valid['pred'] = pd.DataFrame(y_pred[len(y_pred)-nPeriods:,:].flatten(), index=valid.index, dtype=np.float32)

print('Train Mean Absolute Error:', mean_absolute_error(valid['count'], valid['pred']))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(valid['count'], valid['pred'])))

plot_ts(train,valid)



