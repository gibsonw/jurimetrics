
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
tb_jurimetrics_adj <- tb_jurimetrics_adj %>% mutate(year = lubridate::year(date),month = lubridate::month(date)) 
tb_jurimetrics_adj <- tb_jurimetrics_adj %>% group_by(subject_decoded,year,month) %>% summarise(tot_proc_month = sum(count)) %>% arrange(subject_decoded,year,month)
#tb_jurimetrics_adj$tot_proc_month


# cria df com assuntos  
df_subject <- tb_jurimetrics_adj[,1] %>% select(subject_decoded) %>% group_by(subject_decoded) %>% distinct()
#  cria df com assuntos
df_subject <- tb_jurimetrics_adj[,1] %>% group_by(subject_decoded) %>% tally() %>% filter(n > 5)
df_subject <- df_subject$subject_decoded


assunto <- "partido político"
t <- run_fits(tb_jurimetrics_adj,assunto)


fnc_eval_err(t$y_pred,t$y,'R-Squared')
fnc_eval_err(t$y_pred,t$y)


t_list <- tibble(assunto,t[1],t[2],t[3],t[4])

assunto <- "1"
t <- run_fits(tb_jurimetrics_adj,assunto)
class(t)

t[1]

t_list <- t_list %>% add_row(assunto,t[1],t[2],t[3],t[4])

#t_list$`t[2]`[[2]][2]

#1:nrow(df_subject[3,])

nrow(df_subject%>%head(5))

nrow(df_subject%>%head(5))

df_subject$subject_decoded[2]

df_subject

for(i in 1:10){
  row <- df_subject[i]
  #print(row)
  run_fits(tb_jurimetrics_adj,row)
  # do stuff with row
}


run_fits(tb_jurimetrics_adj,assunto)

apply(df_subject, 1, run_fits())




dim(distinct(tb_jurimetrics_adj, subject_decoded))

for(i in 1:nrow(d)) {
  row <- d[i,]
  print(row)
  # do stuff with row
}

d<- tibble(
  name = letters[1:4], 
  plate = c('P1', 'P2', 'P3','P4'), 
  value1 = c(1:4),
  value2 = c(1:4)*100
)


f <- function(x, output) {
  wellName <- x[1]
  plateName <- x[2]
  wellID <- 1
  print(paste(wellID, x[3], x[4], sep=","))
  cat(paste(wellID, x[3], x[4], sep=","), file= output, append = T, fill = T)
}

apply(d, 1, f, output = 'outputfile')











#tb_jurimetrics_dateini_dateend_by_subject <- tb_jurimetrics %>% group_by(subject_decoded) %>% summarise(date_ini = min(date),date_end = max(date))


teste <- tb_jurimetrics %>%filter(subject_decoded == "1") %>% group_by(subject_decoded,year,month) %>% summarise(
  tot_proc_month = sum(count)) %>% arrange(year,month,subject_decoded)


tb_jurimetrics_by_year_month <- tb_jurimetrics %>% group_by(subject_decoded,year,month)


tb_jurimetrics_by_year_month <- tb_jurimetrics_by_year_month %>% summarise(
  tot_proc_month = sum(count)) %>% arrange(year,month,subject_decoded)








by_subject_decoded_date <- tb_jurimetrics %>% group_by(subject_decoded,judgmentDate)

tot_proc_dia <- by_subject_decoded_date %>% summarise(
  tot_proc_dia = sum(count)) %>% arrange(subject_decoded,judgmentDate)


tot_proc_dia <- tot_proc_dia %>%
  pivot_wider(names_from = subject_decoded, values_from = tot_proc_dia, values_fill = list(tot_proc_dia = 0))

tot_proc_dia2 <- tot_proc_dia %>% mutate(judgmentDate,  as.Date(judgmentDate)) 

tot_proc_dia2 <- tot_proc_dia %>% mutate(judgmentDate2,  lubridate::dmy(judgmentDate)) 
year()

lubridate::year(tot_proc_dia2[,0,975])

lubridate::dmy(x)

tot_proc_dia$`1`

tot_proc_dia[,1:2]

min(tot_proc_dia$judgmentDate)

max(tot_proc_dia$judgmentDate)

class(as.Date(min(tot_proc_dia$judgmentDate)))

dif_days <- as.Date(max(tot_proc_dia$judgmentDate)) - as.Date(min(tot_proc_dia$judgmentDate))

ts(tot_proc_dia$`1`,frequency = 365, start = as.Date(min(tot_proc_dia$judgmentDate)), end= as.Date(max(tot_proc_dia$judgmentDate)))

ts(tot_proc_dia[,1:2])


as.Date(min(tot_proc_dia$judgmentDate)) - as.Date(max(tot_proc_dia$judgmentDate))


seq(from = as.Date(min(tot_proc_dia$judgmentDate)), to = as.Date(max(tot_proc_dia$judgmentDate)), by = "day")

print( ts(1:10, frequency = 7, start = c(12, 2)), calendar = TRUE)

class(livestock)

fits(livestock, show.sec.graph = F)

tot_proc_dia


fits(ts(tot_proc_dia[,1:2]))



