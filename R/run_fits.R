
#devtools::install_github('filipezabala/jurimetrics', force=T)
#install.packages('lubridate')

library(tidyverse)
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
write.table(tb_jurimetrics_adj , file = "C:\\PUCRS\\Especialização\\Jurimetrics\\jurimetrics\\data\\tb_jurimetrics_adj.csv")


col_name <- colnames(tb_jurimetrics_adj)

'subject_decoded'        date       count

#write_excel_csv2(tb_jurimetrics_adj, "C:\\PUCRS\\Especialização\\Jurimetrics\\jurimetrics\\data\\tb_jurimetrics_adj.csv", na = "", append = FALSE,col_names = col_name, delim = ";", quote_escape = "double")


tb_jurimetrics_adj <- tb_jurimetrics_adj %>% mutate(year = lubridate::year(date),month = lubridate::month(date)) 
tb_jurimetrics_adj <- tb_jurimetrics_adj %>% group_by(subject_decoded,year,month) %>% summarise(tot_proc_month = sum(count)) %>% arrange(subject_decoded,year,month)
#tb_jurimetrics_adj$tot_proc_month



# cria df com assuntos  
df_subject <- tb_jurimetrics_adj[,1] %>% select(subject_decoded) %>% group_by(subject_decoded) %>% distinct()
#  cria df com assuntos
df_subject <- tb_jurimetrics_adj[,1] %>% group_by(subject_decoded) %>% tally() %>% filter(n > 12)
df_subject <- df_subject$subject_decoded

l <- list() 

for (i in 1:length(df_subject)) {
  ts_1 <- tb_jurimetrics_adj %>% filter(subject_decoded == df_subject[i])
  ts(ts_1[,4])
  t <- fits(ts(ts_1[,4]),train = 0.85,show.main.graph = F,show.sec.graph = F) 
  
  l[[df_subject[[i]]]] <- t
}

i <- 272
ts_1 <- tb_jurimetrics_adj %>% filter(subject_decoded == df_subject[i])
ts(ts_1[,4])
t <- fits(ts(ts_1[,4]),train = 0.85,show.main.graph = F,show.sec.graph = F) 

