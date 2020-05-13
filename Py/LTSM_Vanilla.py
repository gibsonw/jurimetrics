
import pandas as pd
import numpy  as np
import datetime as dt
import string
import math

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
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



pathData =  "C:\\PUCRS\\Especialização\\Jurimetrics\\jurimetrics\\data"
pathFileDaySubject = pathData+'\\tb_jurimetrics_adj.csv'
#pathFileDaySubject =  "https://raw.githubusercontent.com/gibsonw/jurimetrics/master/data/count_day_subject.csv"

date_cols = ['date']
df_index = ['date']
df_jurimetric_subject = pd.read_csv(pathFileDaySubject,sep=";",encoding='UTF-8', index_col=df_index,parse_dates=['date'])
df_jurimetric_subject_bymonth = df_jurimetric_subject.groupby(['subject_decoded']).resample('MS').sum()

df_jurimetric_subject_bymonth.loc[l_subject_decoded[4]].shape[0]

assunto = l_subject_decoded[4]
rmse,df_train,df_valid = run_LTSM(assunto,df_jurimetric_subject_bymonth)


l_subject_decoded = df_jurimetric_subject_bymonth.unstack().index

for i in range(0,50):
    assunto = l_subject_decoded[i]
    qtdd_reg = df_jurimetric_subject_bymonth.loc[l_subject_decoded[i]].shape[0]
    print ('Assunto : {} , Index : {}, Quantidade de Registros {}'.format(assunto,i,qtdd_reg))
    if (qtdd_reg > 70):
        #plot_ts(df_jurimetric_subject_bymonth.loc[assunto],p_assunto=assunto,p_rmse=.0)
        rmse,df_train,df_valid = run_LTSM(assunto,df_jurimetric_subject_bymonth)
        plot_ts(df_train,df_valid,p_assunto=assunto,p_rmse=rmse)





#ds = df_jurimetric_subject_bymonth.loc[assunto].values
#ds.shape



ds[0:training_data_len,:]
df_jurimetric_subject_bymonth.loc[assunto][:training_data_len]

df_jurimetric_subject_bymonth.loc[assunto].values[0:training_data_len,:].shape

ds[0:training_data_len,:]

    train_data = sc_ds[0:training_data_len,:]

    train = df_jurimetric_subject_bymonth.loc[assunto][:training_data_len] # dataset original


# plot the pathData

train = df_jurimetric_subject_bymonth.loc[assunto][:training_data_len] # dataset original
valid = pd.DataFrame(df_jurimetric_subject_bymonth.loc[assunto][training_data_len:],dtype=np.float32)
valid['pred'] = pd.DataFrame(y_pred.flatten(), index=valid.index, dtype=np.float32)

plot_ts(train,valid)


