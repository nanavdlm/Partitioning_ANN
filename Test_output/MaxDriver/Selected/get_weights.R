
remove(list=ls())
{library(tidyverse, quietly = T)
  library(hms)
  library(REddyProc)}

Sys.setenv(TZ = "GMT")
user="rdaelman"
wd=dirname(rstudioapi::getSourceEditorContext()$path)

setwd(wd)

output<- paste0("C:/Users/rdaelman/OneDrive - UGent/CongoFlux/Analyses/Final_EC_output/")

is.outlier <- function (x) {
  x < quantile(x, .25, na.rm=T) - 1.5 * IQR(x, na.rm=T) |
    x > quantile(x, .75, na.rm=T) + 1.5 * IQR(x, na.rm=T)
}
#--- Load data
lst<-list.files(paste0(wd, '/IW/'), full.names = T)
colnames1<-c('RadDaily','RadDailySeas','SW_IN_POT','SW_IN_f','dSW_IN_POT','Tair_f','VPD_f','SWC_f','WS_f','cos_wd','sin_wd','GPPDailyprov')
colnames2<-c('cos_doy','sin_doy','Tair_f','TS_f','SWC_f','WS_f','cos_wd','sin_wd','NeeNightime')
colnames3<-c('SW_IN_f')
list<-lst%>%data.frame(file=.)%>%mutate(name=substr(file, 104, 117), Year=substr(name, 1,4), Model=as.numeric(substr(name, 6,6)), Predictor=as.numeric(substr(name, 9,9)))
lst1<-list%>%filter(Predictor==1)
lst2<-list%>%filter(Predictor==2)
lst3<-list%>%filter(Predictor==3)

Pred1<-read.csv(lst1$file[1], header = F, sep = ",", col.names = colnames1)%>%mutate(Model=lst1$Model[1], Year=lst1$Year[1])
Pred2<-read.csv(lst2$file[1], header = F, sep = ",", col.names = colnames2)%>%mutate(Model=lst2$Model[1], Year=lst2$Year[1])
Pred3<-read.csv(lst3$file[1], header = F, sep = ",", col.names = colnames3)%>%mutate(Model=lst3$Model[1], Year=lst3$Year[1])
for (i in 2:nrow(lst1)){
  Pred1<-rbind(Pred1, read.csv(lst1$file[i], header = F, sep = ",", col.names = colnames1)%>%mutate(Model=lst1$Model[i], Year=lst1$Year[i]))
  Pred2<-rbind(Pred2, read.csv(lst2$file[i], header = F, sep = ",", col.names = colnames2)%>%mutate(Model=lst2$Model[i], Year=lst2$Year[i]))
  Pred3<-rbind(Pred3, read.csv(lst3$file[i], header = F, sep = ",", col.names = colnames3)%>%mutate(Model=lst3$Model[i], Year=lst3$Year[i]))
}


####-------------------------------------------####
####-------- feature importance of ANN --------####
####-------------------------------------------####

lst<-list.files(paste0(wd, '/Perm/'), full.names = T)
Perm<-read.csv(paste0(wd, '/header_perm.csv'))
for (i in lst){
  Perm<-cbind(Perm, read.csv(i,header=F))
}
colnames(Perm)<-c('Variable', 'For', 'Model1_2022', 'Model2_2022','Model3_2022','Model4_2022','Model5_2022','Model1_2023', 'Model2_2023','Model3_2023','Model4_2023','Model5_2023' )

write.csv(Perm, paste0(output, 'Perm.csv'), row.names = F)

lst<-list.files(paste0(wd, '/Perf/'), full.names = T)
Perf<-read.csv(lst[1], header=F)
Perf<-Perf%>%mutate(Year=lst[1]%>%substr(106,109), Model=lst[1]%>%substr(111,111))
for (i in 2:length(lst)){
  Perf<-rbind(Perf, read.csv(lst[i], header=F)%>%mutate(Year=lst[i]%>%substr(106,109), Model=lst[i]%>%substr(111,111)))
}
colnames(Perf)<-c('r','RMSE','MAE' , 'm', 'Year', 'Model')
write.csv(Perf, paste0(output, 'Permf.csv'), row.names = F)
