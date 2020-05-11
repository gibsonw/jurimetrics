
import Feriados_Brasil as fb
import pandas as pd
import datetime as dt
import string
import unicodedata
import codecs



pathData =  "C:\\PUCRS\\Especialização\\Jurimetrics\\jurimetrics\\data"
pathFileDaySubject = pathData+'\\count_day_subject.csv'
#pathFileDaySubject =  "https://raw.githubusercontent.com/gibsonw/jurimetrics/master/data/count_day_subject.csv"
df_count_day_subject = pd.read_csv(pathFileDaySubject,sep=",",encoding='UTF-8')

del df_count_day_subject['Unnamed: 0']

df_count_day_subject['day_name'] = pd.to_datetime(df_count_day_subject['judgmentDate']).dt.day_name()
df_count_day_subject['day_week'] = pd.to_datetime(df_count_day_subject['judgmentDate']).dt.dayofweek

l_feriados = fb.Feriados_Brasil()


lista_feriados = l_feriados.holidays(df_count_day_subject['judgmentDate'].min(), df_count_day_subject['judgmentDate'].max())
lista_weekends = pd.to_datetime(df_count_day_subject[df_count_day_subject['day_week'] >= 5]['judgmentDate'].to_list()).drop_duplicates()
lista_feriados_weekends = lista_feriados.append(lista_weekends)
df_count_day_subject['work_day'] = df_count_day_subject['judgmentDate'].apply(lambda x: x not in lista_feriados_weekends)
df_count_day_subject['subject_decoded'] = df_count_day_subject['subject'].apply(lambda x: codecs.decode(str(x).lower(), 'unicode_escape').encode('ISO-8859-1').decode('UTF-8'))

del df_count_day_subject['subject']

#df_count_day_subject.to_csv('C:\\PUCRS\\Especialização\\Jurimetrics\\jurimetrics\\data\\count_day_subject_decoded.csv', index=False)




import chart_studio.plotly as py
#import plotly.chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
import plotly.express as px

import matplotlib as plt
import seaborn as sns


fig = px.line(df_count_day_subject, x='judgmentDate', y='count')
fig.update_xaxes(rangeslider_visible=True)

fig.show()



sns.set(style="darkgrid")
# Plot the responses for different events and regions
sns.lineplot(x="judgmentDate", y="count",
             hue="subject_decoded", 
             data=df_count_day_subject)

