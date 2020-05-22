'''
# data_jurimetrics.py
import pandas as pd
import numpy  as np
import math
import os
from scipy.stats import normaltest,kurtosis,skew,probplot
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import matplotlib.pyplot as plt
'''

def get_data_jurimetrics(frequency='MS'):
    pathData =  "C:\\PUCRS\\Especialização\\Jurimetrics\\data"
    pathFileDaySubject = pathData+'\\tb_jurimetrics_adj.csv'
    #pathFileDaySubject =  "https://raw.githubusercontent.com/gibsonw/jurimetrics/master/data/count_day_subject.csv"

    date_cols = ['date']
    df_index = ['date']
    df_jurimetric_subject = pd.read_csv(pathFileDaySubject,sep=";",encoding='UTF-8', index_col=df_index,parse_dates=['date'])

    df = df_jurimetric_subject.groupby(['subject_decoded']).resample(frequency).sum()
    return (df)

def get_data_jurimetrics_top_subjects(frequency='MS',percentile=.8,cutOffDate='2014-01-01'):
    df = get_data_jurimetrics(frequency='MS')
    df_subject_most_important = df
    df_subject_most_important = df_subject_most_important.reset_index()
    df_subject_most_important = df_subject_most_important[df_subject_most_important['date'] >= cutOffDate ]
    mask = pd.DataFrame(df_subject_most_important.groupby('subject_decoded')['count'].sum() > df_subject_most_important.groupby('subject_decoded')['count'].sum().quantile(percentile))
    mask = mask.reset_index()
    mask = mask.set_index(['count'])
    list_subjects = mask.loc[True]['subject_decoded'].to_list()
    df_subject = df.loc[list_subjects]
    return (df_subject,list_subjects)



def get_data_measures_subjects(dfSubject):
    
    lSubjects = dfSubject.index.get_level_values(0).unique().to_list()

    df_subject_measures = pd.DataFrame(lSubjects,columns=['subject'])
    df_subject_measures = df_subject_measures.set_index(['subject'])
    df_subject_measures['normaltest_p'] = 0.0
    df_subject_measures['normaltest'] = 0.0
    df_subject_measures['adfuller_adf'] = 0.0
    df_subject_measures['adfuller_pvalue'] = 0.0
    df_subject_measures['kurtosis'] = 0.0
    df_subject_measures['skew'] = 0.0
    for index, row in df_subject_measures.iterrows():
        #print(index)
        #Statistical Normality Test
        stat, p = normaltest(dfSubject.loc[index])
        df_subject_measures.loc[index]['normaltest_p'] = p
        df_subject_measures.loc[index]['normaltest'] = stat
        '''
        Stationarity
        In statistics, the Dickey–Fuller test tests the null hypothesis that a unit root is present in an autoregressive model. The alternative hypothesis is different depending on which version of the test is used, but is usually stationarity or trend-stationarity.
        Stationary series has constant mean and variance over time. Rolling average and the rolling standard deviation of time series do not change over time.
        Dickey-Fuller test
        Null Hypothesis (H0): It suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.
        Alternate Hypothesis (H1): It suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.
        p-value > 0.05: Accept the null hypothesis (H0), the data has a unit root and is non-stationary.
        p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
        '''
        df_subject_measures.loc[index][['adfuller_adf','adfuller_pvalue']] = adfuller(dfSubject.loc[index],autolag='AIC',)[:2]
        df_subject_measures.loc[index]['kurtosis'] = kurtosis(dfSubject.loc[index])
        df_subject_measures.loc[index]['skew'] = skew(dfSubject.loc[index])

    return df_subject_measures



def dash_jurimetrics_subject(dfSubject,subject):
    df = dfSubject.loc[subject]
    dfs = df.reset_index()
    fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(12,12))
    sns.distplot(df, label=subject,bins=30,ax=axs[0,0])
    #ax = ax.set_title(subject)
    #sns.lmplot(x='date', y='count',data=df.reset_index(),fit_reg=False,)
    #sns.scatterplot(x='date', y='count',data=dfs,ax=axs[1])
    sns.lineplot(x='date', y='count', data=dfs,dashes=True,markers=True, marker='o',ci=95,ax=axs[0,1])
    sns.boxplot(data=df,ax=axs[1,0])    
    '''
    Calculate quantiles for a probability plot, and optionally show the plot.
    Generates a probability plot of sample data against the quantiles of a specified theoretical distribution (the normal distribution by default). probplot optionally calculates a best-fit line for the data and plots the results using Matplotlib or a given plot function.
    '''
    osm, osr = probplot(df['count'],plot=axs[1,1])
    fig.tight_layout(pad=3.0)
 
    df1 = get_data_jurimetrics(frequency='D')
    df1.reset_index(inplace=True)
    df1.set_index('date',inplace=True)

    fig = plt.figure(figsize=(12,12))
    fig.subplots_adjust(hspace=.8)
    ax1 = fig.add_subplot(5,1,1)
    ax1.plot(df1[df1['subject_decoded']==subject].resample('D').mean(),linewidth=1)
    ax1.set_title('Média Diária')
    ax1.tick_params(axis='both', which='major')

    ax2 = fig.add_subplot(5,1,2, sharex=ax1)
    ax2.plot(df1[df1['subject_decoded']==subject].resample('W').mean(),linewidth=1)
    ax2.set_title('Média Semanal')
    ax2.tick_params(axis='both', which='major')

    ax3 = fig.add_subplot(5,1,3, sharex=ax1)
    ax3.plot(df1[df1['subject_decoded']==subject].resample('M').mean(),linewidth=1)
    ax3.set_title('Média Mensal')
    ax3.tick_params(axis='both', which='major')

    ax4 = fig.add_subplot(5,1,4, sharex=ax1)
    ax4.plot(df1[df1['subject_decoded']==subject].resample('Q').mean(),linewidth=1)
    ax4.set_title('Média Trimestral')
    ax4.tick_params(axis='both', which='major')

    ax5 = fig.add_subplot(5,1,5, sharex=ax1)
    ax5.plot(df1[df1['subject_decoded']==subject].resample('A').mean(),linewidth=1)
    ax5.set_title('Média Anual')
    ax5.tick_params(axis='both', which='major')


def trend_jurimetrics_subject(subject):
    df1 = get_data_jurimetrics(frequency='D')
    df1 = df1.loc[subject]
    df1.reset_index(inplace=True)

    df1['year'] = df1['date'].apply(lambda x: x.year)
    df1['quarter'] = df1['date'].apply(lambda x: x.quarter)
    df1['month'] = df1['date'].apply(lambda x: x.month)
    df1['day'] = df1['date'].apply(lambda x: x.day)

    df1.sort_values('date', inplace=True, ascending=True)

    plt.figure(figsize=(14,8))
    plt.subplot(2,2,1)
    df1.groupby('year')['count'].agg('mean').plot()
    plt.xlabel('')
    plt.title('Média de processos por Ano')

    plt.subplot(2,2,2)
    df1.groupby('quarter')['count'].agg('mean').plot()
    plt.xlabel('')
    plt.title('Média de processos por Quarter')

    plt.subplot(2,2,3)
    df1.groupby('month')['count'].agg('mean').plot()
    plt.xlabel('')
    plt.title('Média de processos por Mês')

    plt.subplot(2,2,4)
    df1.groupby('day')['count'].agg('mean').plot()
    plt.xlabel('')
    plt.title('Média de processos por Dia')


    pd.pivot_table(df1.loc[df1['year'] > 2009], values = "count", 
                columns = "year", index = "month").plot(subplots = True, figsize=(12, 12), layout=(2, 4), sharey=True)

def stationarity_jurimetrics_subject(subject,lag=12,frequency='D'):
    df1 = get_data_jurimetrics(frequency='D')
    timeseries = df1['count'].loc[subject].resample(frequency).sum()

    rolmean = timeseries.rolling(window=lag).mean()
    rolstd = timeseries.rolling(window=lag).std()
    
    plt.figure(figsize=(14,5))
    sns.despine(left=True)
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best'); plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    print ('<Results of Dickey-Fuller Test>')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


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

