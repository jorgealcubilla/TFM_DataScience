## -------------------------------------------------------------------------
## Clustering for model prediction error:  kMEANS CLUSTERING 
  
## -------------------------------------------------------------------------

## -------------------------------------------------------------------------
##### Libraries required ####

library(dplyr)
setwd("./")

## -------------------------------------------------------------------------
##### Data_frame loading #####

testData=read.csv("test_r",stringsAsFactors = FALSE)

## -------------------------------------------------------------------------
##### Data_frame:Basic information #####

str(testData)
head(testData)
summary(testData)

## -------------------------------------------------------------------------
##### Variables preprocessing #####

#testData$cam = as.factor(testData$cam)
testData$error_th10 = as.factor(testData$error_th10)

# Features relevant for k-means clustering:
clusterData = testData[c('error','ntrue','overlapping','bright','AvgDistance')]

## -------------------------------------------------------------------------
##### K-Means Clustering  #####

# Variables normalization:
clusterDataScaled=scale(clusterData)

# Number of clusters (based on "Elbow method", see ANNEX below)
NUM_CLUSTERS=15
# k-means tends to converge to local minimums, being highly sensitive to initialization conditions.
set.seed(1234) 

Model=kmeans(clusterDataScaled,NUM_CLUSTERS)

clusterData$Clusters=Model$cluster
clusterData$img =testData$img
clusterData$cam =testData$cam

## -------------------------------------------------------------------------
##### Analisis of clusters ####

# Let´s see clusters features:
table(clusterData$Clusters)

aggregate(clusterData[,-7:-8], by = list(clusterData$Clusters), median)

clusterData %>% 
  group_by(Clusters) %>% 
    summarise(maxError=max(error), minError=min(error), maxTrue = max(ntrue), minTrue=min(ntrue), 
             maxOvlap=max(overlapping), minOvlap = min(overlapping), maxBright=max(bright), 
            minBright = min(bright),maxDispers = max(AvgDistance), minDispers = min(AvgDistance))

# And let´s see the prediction error levels split by cluster:
table(clusterData$Clusters,testData[,13])
# The clusters# that more clearly represent "high predicition error" profiles are: 1, 6, 9, 14 and 15
# their main features are as follows:

####Summary:

## Cluster# 14:
clusterData[clusterData$Clusters == "14",]
#Highest positive errors since it combines high values for all the metrics 
#(brightness which is specially high, groundtruth, overlapping and dispersion) 

## Clusters# 6:
clusterData[clusterData$Clusters == "6",]
#Highest negative errors
#High brightness and specially high dispersion  
#Moderate ground_truth, low overlapping 
 
## Cluster# 9: 
clusterData[clusterData$Clusters == "9",]
#Negative errors
#High brightness, low dispersion
#Moderate ground_truth, low overlapping

## Cluster# 15: 
clusterData[clusterData$Clusters == "15",]
#Positive errors
#Moderate brightness and dispersion
#High ground_truth and overlapping

## Cluster# 1: 
clusterData[clusterData$Clusters == "1",]
#Positive errors, from moderate to extremely high
#High brightness and dispersion
#Highest ground_truth and overlapping

## -------------------------------------------------------------------------
##### Conclusions #####

# In order to detect images with "high predicition error", we have selected the clusters
# that more clearly represent this kind of profile, which are clusters#: 1, 6, 9, 14 and 15
#Their features are different to each other (see above) making them representative of 
#specific profiles for high levels of predicition error.

# See "analysis_for_improvemet" notebooks of this project for further analysis.

## -------------------------------------------------------------------------
##### ANNEX: Number of clusters selection (Elbow Method) #####

Intra <- (nrow(clusterData[,-6:-7])-1)*sum(apply(clusterData[,-7:-8],2,var))
for (i in 2:20) Intra[i] <- sum(kmeans(clusterData[,-7:-8], centers=i)$withinss)
plot(1:20, Intra, type="b", xlab="Number of Clusters", ylab="Mean distance to centroids")

# After having a look to the clusters resulting from the "elbow" area (from 5 to 15 clusters), 
# the most meaningful information is provided by 15 clusters

## -------------------------------------------------------------------------
##### Saving clusters data #####
write.csv(clusterData[, c('img','Clusters')], 'clustersR.csv')

## -------------------------------------------------------------------------

