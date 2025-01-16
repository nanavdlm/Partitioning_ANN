library(tidyverse)
wd=dirname(rstudioapi::getSourceEditorContext()$path)
input<-paste0(wd, "/CD_Ygb/")
q<-list.files(input)
q<-q[-length(q)]
ANN<-read.csv(paste0(input, q[1]), header=TRUE, sep=",")
for (i in 2:length(q)){
  ANN<-rbind(ANN, read.csv(paste0(input, q[i]), header=TRUE, sep=","))
}
write.csv(ANN, paste0(wd, "/CD_Ygb/CD_Ygb.csv"), row.names=FALSE)
