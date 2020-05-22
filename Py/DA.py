import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')
from time import time
import matplotlib.ticker as tkr
from scipy import stats
from scipy.stats import normaltest,kurtosis,skew,probplot

from statsmodels.tsa.stattools import pacf,adfuller

from . import get_data_jurimetrics 
from . import get_data_jurimetrics_top_subjects 
from . import get_data_measures_subjects
from . import dash_jurimetrics_subject
from . import trend_jurimetrics_subject
from . import stationarity_jurimetrics_subject

df = get_data_jurimetrics(frequency='MS')

df_subject, list_subjects = get_data_jurimetrics_top_subjects()

df_subject['count'].sum()/df['count'].sum()
df_subject.describe()

df_subject_measures = get_data_measures_subjects(df_subject)

df_subject_measures.to_csv('C:\\PUCRS\\Especialização\\Jurimetrics\\data\\df_subject_measures.csv',sep=";",index=True)

dash_jurimetrics_subject(df_subject,'adicional de horas extras')
 
subject = 'adicional de horas extras'
subject = 'alimentos'
subject = 'obrigações' 
subject = 'latrocínio'

trend_jurimetrics_subject(subject)
 
stationarity_jurimetrics_subject(subject,lag=12,frequency='MS')

