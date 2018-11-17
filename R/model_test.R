## -------------------------------------------------------------------------
## SCRIPT: Clustering for model error
## -------------------------------------------------------------------------

## -------------------------------------------------------------------------

##### 1. Bloque de inicializacion de librerias #####

if(!require("dummies")){
  install.packages("dummies")
  library("dummies")
}

#setwd("C:/Users/JORGE/Desktop/DATA SCIENCE MASTER/R MASTER/STATS_MODELIZACION/APRENDIZAJE NO SUPERVISADO")
setwd("./")


## -------------------------------------------------------------------------
##       PARTE 2: CLUSTERING kMEANS
## -------------------------------------------------------------------------

##### 8. Bloque de carga de Datos #####

testData=read.csv("test_r",stringsAsFactors = FALSE)

## -------------------------------------------------------------------------

##### 9. Bloque de revisión basica del dataset #####

str(testData)
head(testData)
summary(testData)

## -------------------------------------------------------------------------

##### 10. Bloque de tratamiento de variables #####

#testData$date=as.Date(testData$date)
#testData$time=as.Date(testData$time, format = '%H:%M:%S')
#testData$date_time=as.Date(testData$date_time,format='%d/%b/%Y:%H:%M:%S')

testData$cam = as.factor(testData$cam)
testData$error_seg = as.factor(testData$error_seg)
testData$re_error_seg = as.factor(testData$re_error_seg)

clusterData = testData[c('ntrue','error','mean','AvgDistance','overlapping','std')]

## -------------------------------------------------------------------------

##### 11. Bloque de Segmentación mediante Modelo RFM 12M  #####

clusterDataScaled=scale(clusterData)
#Normalización

NUM_CLUSTERS=15
set.seed(1234)
Modelo=kmeans(clusterDataScaled,NUM_CLUSTERS)

clusterData$Clusters=Modelo$cluster
clusterData$img =testData$img
clusterData$cam =testData$cam

table(clusterData$Clusters)

aggregate(clusterData[,-8:-9], by = list(clusterData$Clusters), median)

table(clusterData$Clusters,testData[,20])
## -------------------------------------------------------------------------

##### 12. Bloque de Ejericio  #####

## Elegir el numero de clusters

## -------------------------------------------------------------------------

##### 13. Bloque de Metodo de seleccion de numero de clusters (Elbow Method) #####

Intra <- (nrow(clusterData)-1)*sum(apply(clusterData,2,var))
for (i in 2:20) Intra[i] <- sum(kmeans(clusterData, centers=i)$withinss)
plot(1:20, Intra, type="b", xlab="Numero de Clusters", ylab="Suma de Errores intragrupo")

##### 11. Error test #####

table(clusterData$Clusters,testData[,20])

error_segments<-c("High","Low","Low", "Low","Low","High","Low","Low","Low","Low","Low","Low","Low","High","Low")

Error=(length(testData[,20])-sum(error_segments[clusterData$Clusters]==testData[,20]))/length(testData[,20])
## (Total_registros - sum(coincidencias entre "estado real" y asignación del cluster)/Total_registros
Error

clusterData[clusterData$Clusters == "14",]
