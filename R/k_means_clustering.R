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
clusterData = testData[c('ntrue','error','AvgDistance','overlapping')]

## -------------------------------------------------------------------------
##### K-Means Clustering  #####

# Variables normalization:
clusterDataScaled=scale(clusterData)

# Number of clusters (based on "Elbow method", see ANNEX below)
NUM_CLUSTERS=6
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

aggregate(clusterData[,-6:-7], by = list(clusterData$Clusters), median)

clusterData %>% 
  group_by(Clusters) %>% 
  summarise(maxTrue = max(ntrue), minTrue=min(ntrue), 
            maxError=max(error), minError=min(error), maxDispers = max(AvgDistance), 
            minDispers = min(AvgDistance), maxOvlap=max(overlapping), minOvlap = min(overlapping))

# And let´s see the prediction error levels split by cluster:
table(clusterData$Clusters,testData[,12])
# Clusters# 1, 3 and 6 have higher percentage of "High" level than "Low" level
# Thus, they will be categorized as clusters of "high prediction error" and the rest as 
# clusters of "low prediction error"

####Summary:
 
## Cluster# 1: 
# Predicition error level: High (positive error)
# Ground_truth: High (ntrue>50)
# Overlapping: High (>20) 
# Dispersion: Medium (median AvrgDistance = 152)
#It is not exclusive of an specific camera (very important)
clusterData[clusterData$Clusters == "1",]

## Cluster# 2: 
# Predicition error level: Low (combination of positive and negative errors)
# Ground_truth: Low (ntrue<50)
# Overlapping: Low (20<) 
# Dispersion: Low (AvrgDistance < 142)
#It is not exclusive of an specific camera (very important)
clusterData[clusterData$Clusters == "2",]

## Cluster# 3: 
# Predicition error level: High (negative error)
# Ground_truth: Low (ntrue<35)
# Overlapping: Low (<10) 
# Dispersion: Low (AvrgDistance < 185)
#It is not exclusive of an specific camera (very important)
clusterData[clusterData$Clusters == "3",]

## Cluster# 4: 
# Predicition error level: Low (combination of positive and negative errors)
# Ground_truth: Low (ntrue<50)
# Overlapping: Low (<20) 
# Dispersion: Low (AvrgDistance < 185)
#It is not exclusive of an specific camera (very important)
clusterData[clusterData$Clusters == "4",]

## Cluster# 5: 
# Predicition error level: Low (positive errors)
# Ground_truth: High (ntrue>35)
# Overlapping: Low (<20) 
# Dispersion: Medium (median AvrgDistance = 164)
#It is not exclusive of an specific camera (very important)
clusterData[clusterData$Clusters == "5",]

## Cluster# 6: 
# Predicition error level: High (negative error)
# Ground_truth: Low (ntrue<50)
# Overlapping: Low (<20) 
# Dispersion: High (265>AvrgDistance>175)
#It is not exclusive of an specific camera (very important)
clusterData[clusterData$Clusters == "6",]

## -------------------------------------------------------------------------
##### Conclusions #####

# The "high prediction error" clusters# are: 1, 3 and 6
# Their features are summarised above

## -------------------------------------------------------------------------
##### ANNEX: Number of clusters selection (Elbow Method) #####

Intra <- (nrow(clusterData[,-6:-7])-1)*sum(apply(clusterData[,-6:-7],2,var))
for (i in 2:20) Intra[i] <- sum(kmeans(clusterData[,-6:-7], centers=i)$withinss)
plot(1:20, Intra, type="b", xlab="Number of Clusters", ylab="Mean distance to centroids")

# After having a look to the clusters resulting from the "elbow" area (from 5 to 15 clusters), 
# the most meaningful information is provided by 6 clusters


