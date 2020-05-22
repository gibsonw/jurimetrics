
#devtools::install_github('filipezabala/jurimetrics', force=T)
#install.packages('lubridate')

library(tidyverse)
library("readr")

#library(lubridate)
#library(plyr)
#library(dplyr)

#library(jurimetrics)

#
#df_jurimetrics <- read.table('https://raw.githubusercontent.com/gibsonw/jurimetrics/master/data/count_day_subject_decoded.csv',
#                             sep = ',',
#                             encoding = 'UTF-8',
#                             head = T)


df_jurimetrics <- read.table("C:\\PUCRS\\Especialização\\Jurimetrics\\jurimetrics\\data\\count_day_subject_decoded.csv",
                             sep = ',',
                             encoding = 'UTF-8',
                             head = T)

class(df_jurimetrics)

tb_jurimetrics <- df_jurimetrics %>% tibble()  

class(tb_jurimetrics)

tb_jurimetrics <- tb_jurimetrics %>% mutate(date =  as.Date(judgmentDate)) 
tb_jurimetrics <- tb_jurimetrics %>% mutate(year = lubridate::year(date), month = lubridate::month(date)) 



# separa o df por assunto e cria duas colunas com data inicial e data final
tb_subject_decoded_ini_end_col <- tb_jurimetrics %>% group_by(subject_decoded) %>% summarise(date_ini = min(date),date_end = max(date))
# cria df com valores zerados em todos os dias entre a data inicial e data final
tb_subject_decoded_ini_end_row <- tb_subject_decoded_ini_end_col %>% group_by(subject_decoded) %>% do(data.frame(subject_decoded=.$subject_decoded, date=seq(.$date_ini,.$date_end,by="1 month"))) %>% add_column(count = 0)

# concatena o df original e o df com as datas com valores zerados 
tb_jurimetrics_adj = bind_rows(tb_jurimetrics %>% select(subject_decoded,date,count), tb_subject_decoded_ini_end_row)
#write_delim(tb_jurimetrics_adj , "C:\\PUCRS\\Especialização\\Jurimetrics\\jurimetrics\\data\\tb_jurimetrics_adj.csv", delim = ";")



tb_jurimetrics_adj <- tb_jurimetrics_adj %>% mutate(year = lubridate::year(date),month = lubridate::month(date)) 
tb_jurimetrics_adj <- tb_jurimetrics_adj %>% group_by(subject_decoded,year,month) %>% summarise(tot_proc_month = sum(count)) %>% arrange(subject_decoded,year,month)
#tb_jurimetrics_adj$tot_proc_month



# cria df com assuntos  
df_subject <- tb_jurimetrics_adj[,1] %>% select(subject_decoded) %>% group_by(subject_decoded) %>% distinct()
#  cria df com assuntos
df_subject <- tb_jurimetrics_adj[,1] %>% group_by(subject_decoded) %>% tally() %>% filter(n > 20)
df_subject <- df_subject$subject_decoded

df_subject

l <- list() 
df_mesures <- data.frame()
df_pred_values <- data.frame()
df_aic_values <- data.frame()

for (i in 1:length(df_subject)) {
#for (i in 1:10) {
  ts_1 <- tb_jurimetrics_adj %>% filter(subject_decoded == df_subject[i])
  ts(ts_1[,4])
  t <- fits(ts(ts_1[,4]),train = 0.85,trainPeridos = 12,show.main.graph = F,show.sec.graph = F) 
  
  l[[df_subject[[i]]]] <- t

  
  t$all_fnc_err$subject <- df_subject[i]
  t$all_fnc_err$best_model <- t$best.model
  df_mesures <- rbind(df_mesures,t$all_fnc_err)

  
  tmp_df_pred_values <- data.frame()
  tmp_df_pred_values <- data.frame(t$Y_pred_aa)
  tmp_df_pred_values$y_pred_ets <- data.frame(t$Y_pred_ets)
  tmp_df_pred_values$y_pred_tb  <- data.frame(t$Y_pred_tb)
  tmp_df_pred_values$y_pred_nn  <- data.frame(t$Y_pred_nn)
  tmp_df_pred_values$subject    <- df_subject[i]
  tmp_df_pred_values$best.model <- t$best.model
  df_pred_values <- rbind(df_pred_values,tmp_df_pred_values)
  
  
  tmp_df_aic_values <- data.frame()
  tmp_df_aic_values <- data.frame(t$aic) 
  tmp_df_aic_values$subject    <- df_subject[i]
  tmp_df_aic_values$best.model <- t$best.model
  df_aic_values <- rbind(df_aic_values,tmp_df_aic_values)
  
  }

l

write_delim(df_mesures , "C:\\PUCRS\\Especialização\\Jurimetrics\\jurimetrics\\data\\df_mesures.csv", delim = ";")

df_pred_values$t.Y_pred_aa <- round(df_pred_values$t.Y_pred_aa,6)
df_pred_values$y_pred_ets <- round(df_pred_values$y_pred_ets,6)
df_pred_values$y_pred_tb <- round(df_pred_values$y_pred_tb,6)
df_pred_values$y_pred_nn <- round(df_pred_values$y_pred_nn,6)

write_delim(df_pred_values , "C:\\PUCRS\\Especialização\\Jurimetrics\\jurimetrics\\data\\df_pred_values.csv", delim = ";")
write_delim(df_aic_values , "C:\\PUCRS\\Especialização\\Jurimetrics\\jurimetrics\\data\\df_aic_values.csv", delim = ";")


i <- 187
subject <- 'telefonia'
ts_1 <- tb_jurimetrics_adj %>% filter(subject_decoded == subject)
ts(ts_1[,4])
t2 <- fits(ts(ts_1[,4]),train = 0.85,trainPeridos = 12,show.main.graph = F,show.sec.graph = T) 

t2


t2$all_fnc_err
t2$all_fnc_err$subject <- 'x'
t2$all_fnc_err$best_model <- t2$best.model
df_mesures <- rbind(df_mesures,t2$all_fnc_err)

df_pred_values <- data.frame()

df_pred_values <- data.frame(t2$Y_pred_aa)
df_pred_values$y_pred_ets <- data.frame(t2$Y_pred_ets)
df_pred_values$y_pred_tb <- data.frame(t2$Y_pred_tb)
df_pred_values$y_pred_nn <- data.frame(t2$Y_pred_nn)
df_pred_values$subject <- 'ddd'
df_pred_values <- t2$best.model

  
t2$best.model







t <- "acidente de trabalho"
ts_1 <- tb_jurimetrics_adj %>% filter(subject_decoded == t)
ts(ts_1[,4])
t <- fits(ts(ts_1[,4]),train = 0.85,show.main.graph = T,show.sec.graph = F) 

t$all_fnc_err$pred.nn.rmse
