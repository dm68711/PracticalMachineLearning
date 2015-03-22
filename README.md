---
title: "Machine Learning: Classifying Weightlifting Exercises"
author: "Don Mowbray"
output: html_document
---

<center><h1>Classifying Weightlifting Exercises</h1></center>
<center><h4>Practical Machine Learning, Coursera</h4></center>
<center><h4>Don Mowbray</h4></center>
<center><h4>March 21, 2015</h4></center>

## Overview ##

This report examines the Weight Lifting Exercises (WLE) Dataset (
[citation](http://groupware.les.inf.puc-rio.br/har) ) to predict
weightlifting activity quality from several activity
monitors. Activity quality is represented by the variable *classe*, a
factor with five levels representing how a barbell lifting activity
was conducted:

* according to the specification (Class A), 
* elbows to the front (Class B), 
* lifting halfway (Class C), 
* lowering halfway (Class D)
* hips to the front (Class E).


## Loading and Cleaning the DataSet ##

The following code was used to load and clean the datasets. Cleaning
the dataset involved removing non-predictive columns (user name,
timestamp, etc.) and deleting columns with missing (NA) values.


```r
# load the libraries required for this project
library(AppliedPredictiveModeling)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
## Loading required package: methods
```

```r
library(rpart)
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.4.1 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```


```r
# load the csv training and testing sets
training <- read.csv('~/Downloads/pml-training.csv', na.strings=c("NA", ""), header=TRUE)
testing  <- read.csv('~/Downloads/pml-testing.csv', na.strings=c("NA", ""), header=TRUE)

# remove non-predictive columns
training <- training[,-c(1:7)]
testing  <- testing[,-c(1:7)]  

# Delete columns with missing values
training <-training[,colSums(is.na(training)) == 0]
testing <-testing[,colSums(is.na(testing)) == 0]
```


## Partitioning for Cross Validation ##

*createDataPartition* is used to create a balanced 75/25 split of the
training dataset into training and testing subsamples. Balanced
implies that random sampling occurs within each class to preserve the
overall class distribution of the data. The resulting training subset
will be used to construct our models, and the testing subset will be
employed for cross validation to estimate the accuracy of our
candidate models using data that is independent from the training
subset used to fit our models. Thus, cross validation allows us to
select the most accurate model to apply to the final testing dataset.


```r
set.seed(1234)
inTrain = createDataPartition(training$classe, p = 0.75, list=FALSE)
subTraining = training[inTrain,]
subTesting = training[-inTrain,]
```


## Classification Tree Model ###

First we fit a classification tree against the training subset.


```r
treeModel <- rpart(classe ~ ., data=subTraining, method="class")
print(treeModel, digits=3)
```

```
## n= 14718 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##     1) root 14718 10500 A (0.28 0.19 0.17 0.16 0.18)  
##       2) roll_belt< 130 13483  9310 A (0.31 0.21 0.19 0.18 0.11)  
##         4) pitch_forearm< -34.3 1163     4 A (1 0.0034 0 0 0) *
##         5) pitch_forearm>=-34.3 12320  9300 A (0.24 0.23 0.21 0.2 0.12)  
##          10) magnet_dumbbell_y< 440 10462  7500 A (0.28 0.18 0.24 0.19 0.11)  
##            20) roll_forearm< 124 6480  3840 A (0.41 0.18 0.18 0.17 0.06)  
##              40) magnet_dumbbell_z< -25.5 2237   745 A (0.67 0.2 0.019 0.081 0.031)  
##                80) roll_forearm>=-136 1859   401 A (0.78 0.16 0.02 0.028 0.0043) *
##                81) roll_forearm< -136 378   228 B (0.09 0.4 0.011 0.34 0.16) *
##              41) magnet_dumbbell_z>=-25.5 4243  3090 A (0.27 0.17 0.27 0.21 0.076)  
##                82) yaw_belt>=168 577    89 A (0.85 0.087 0 0.062 0.0052) *
##                83) yaw_belt< 168 3666  2540 C (0.18 0.19 0.31 0.24 0.087)  
##                 166) accel_dumbbell_y>=-40.5 3160  2300 D (0.21 0.21 0.22 0.27 0.094)  
##                   332) pitch_belt< -42.8 382    68 B (0.026 0.82 0.1 0.026 0.024) *
##                   333) pitch_belt>=-42.8 2778  1930 D (0.23 0.12 0.23 0.3 0.1)  
##                     666) roll_belt>=126 652   261 C (0.37 0.017 0.6 0.014 0.0046)  
##                      1332) magnet_belt_z< -324 209     6 A (0.97 0 0.014 0 0.014) *
##                      1333) magnet_belt_z>=-324 443    55 C (0.079 0.025 0.88 0.02 0) *
##                     667) roll_belt< 126 2126  1290 D (0.19 0.16 0.12 0.39 0.13)  
##                      1334) pitch_belt>=1.04 1344  1030 B (0.23 0.24 0.13 0.22 0.19)  
##                        2668) accel_dumbbell_z< 27.5 846   555 A (0.34 0.15 0.2 0.27 0.035)  
##                          5336) yaw_forearm>=-94.7 618   327 A (0.47 0.19 0.22 0.078 0.037)  
##                           10672) magnet_forearm_z>=-68.5 382   102 A (0.73 0.14 0.013 0.1 0.016) *
##                           10673) magnet_forearm_z< -68.5 236   104 C (0.047 0.28 0.56 0.038 0.072) *
##                          5337) yaw_forearm< -94.7 228    51 D (0 0.039 0.15 0.78 0.031) *
##                        2669) accel_dumbbell_z>=27.5 498   274 E (0.034 0.38 0.006 0.13 0.45)  
##                          5338) roll_dumbbell< 53.4 310   139 B (0.052 0.55 0.0097 0.19 0.2) *
##                          5339) roll_dumbbell>=53.4 188    25 E (0.0053 0.09 0 0.037 0.87) *
##                      1335) pitch_belt< 1.04 782   235 D (0.13 0.019 0.11 0.7 0.041) *
##                 167) accel_dumbbell_y< -40.5 506    68 C (0.0099 0.047 0.87 0.038 0.04) *
##            21) roll_forearm>=124 3982  2660 C (0.08 0.18 0.33 0.22 0.18)  
##              42) magnet_dumbbell_y< 292 2330  1200 C (0.096 0.13 0.49 0.15 0.14)  
##                84) magnet_forearm_z< -251 187    36 A (0.81 0.08 0 0.037 0.075) *
##                85) magnet_forearm_z>=-251 2143  1010 C (0.034 0.14 0.53 0.16 0.15) *
##              43) magnet_dumbbell_y>=292 1652  1100 D (0.058 0.25 0.11 0.33 0.25)  
##                86) accel_forearm_x>=-102 1070   708 E (0.052 0.31 0.16 0.14 0.34)  
##                 172) magnet_arm_y>=188 456   208 B (0.018 0.54 0.24 0.096 0.11) *
##                 173) magnet_arm_y< 188 614   300 E (0.078 0.14 0.1 0.17 0.51) *
##                87) accel_forearm_x< -102 582   177 D (0.069 0.13 0.029 0.7 0.079) *
##          11) magnet_dumbbell_y>=440 1858   912 B (0.029 0.51 0.041 0.23 0.19)  
##            22) total_accel_dumbbell>=5.5 1315   452 B (0.04 0.66 0.057 0.022 0.22)  
##              44) roll_belt>=-0.58 1109   246 B (0.048 0.78 0.068 0.026 0.08) *
##              45) roll_belt< -0.58 206     0 E (0 0 0 0 1) *
##            23) total_accel_dumbbell< 5.5 543   145 D (0 0.15 0.0037 0.73 0.11) *
##       3) roll_belt>=130 1235    10 E (0.0081 0 0 0 0.99) *
```

Next we formulate predictions from our testing subset. Then we we use
cross validation to estimate the tree model's accuracy and out of
sample error.


```r
treePredict <- predict(treeModel, subTesting, type = "class")
cmTree <- confusionMatrix(treePredict, subTesting$classe)
```

```
## Loading required namespace: e1071
```

```r
cmTree
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1235  157   16   50   20
##          B   55  568   73   80  102
##          C   44  125  690  118  116
##          D   41   64   50  508   38
##          E   20   35   26   48  625
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7394          
##                  95% CI : (0.7269, 0.7516)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6697          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8853   0.5985   0.8070   0.6318   0.6937
## Specificity            0.9307   0.9216   0.9005   0.9529   0.9678
## Pos Pred Value         0.8356   0.6469   0.6313   0.7247   0.8289
## Neg Pred Value         0.9533   0.9054   0.9567   0.9296   0.9335
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2518   0.1158   0.1407   0.1036   0.1274
## Detection Prevalence   0.3014   0.1790   0.2229   0.1429   0.1538
## Balanced Accuracy      0.9080   0.7601   0.8537   0.7924   0.8307
```

The resulting confusion matrix suggests this model has an accuracy of
0.739. Calculating the out
of sample error as 1 - accuracy, this yields a out of sample error of
26.06 percent.

## Random Forest Model ##

Next we fit a random forest against the training subset. During
preprocessing, principal component analysis (PCA) is used to reduce
the dimensionality of the data (feature reduction) while preserving
the data's essential variance. With the *cv* method parameter, we tune
the training function to use k-folds cross validation. We choose an
*ntree* parameter of 50 to balance model accuracy with computational
performance.


```r
tc <- trainControl(method="cv", number=5, verboseIter=FALSE , preProcOptions="pca")
rfModel <- train(classe ~ ., data=subTraining, method="rf", trControl=tc, ntree=50)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
varImp(rfModel$finalModel, top=20) # show variable importance
```

```
##                         Overall
## roll_belt            1510.34925
## pitch_belt            648.95216
## yaw_belt              847.17762
## total_accel_belt       43.02556
## gyros_belt_x           40.72532
## gyros_belt_y           40.91231
## gyros_belt_z          196.57762
## accel_belt_x           36.37387
## accel_belt_y           43.88806
## accel_belt_z          145.88193
## magnet_belt_x         171.01428
## magnet_belt_y         266.39955
## magnet_belt_z         279.64971
## roll_arm              170.25555
## pitch_arm             106.26960
## yaw_arm               181.58703
## total_accel_arm        57.35252
## gyros_arm_x            58.40700
## gyros_arm_y            79.49423
## gyros_arm_z            21.39828
## accel_arm_x           128.92697
## accel_arm_y            68.50400
## accel_arm_z            56.10444
## magnet_arm_x           99.10971
## magnet_arm_y          106.79952
## magnet_arm_z          112.52877
## roll_dumbbell         337.68691
## pitch_dumbbell         93.69068
## yaw_dumbbell          129.26177
## total_accel_dumbbell  211.60017
## gyros_dumbbell_x       79.77873
## gyros_dumbbell_y      131.99030
## gyros_dumbbell_z       43.54545
## accel_dumbbell_x       88.82517
## accel_dumbbell_y      361.68605
## accel_dumbbell_z      227.13457
## magnet_dumbbell_x     227.94624
## magnet_dumbbell_y     684.83420
## magnet_dumbbell_z     676.22510
## roll_forearm          590.75975
## pitch_forearm         964.04495
## yaw_forearm           106.62725
## total_accel_forearm    42.32006
## gyros_forearm_x        25.95104
## gyros_forearm_y        65.10682
## gyros_forearm_z        41.93177
## accel_forearm_x       291.04692
## accel_forearm_y        83.63184
## accel_forearm_z       183.23407
## magnet_forearm_x       99.80298
## magnet_forearm_y      101.73320
## magnet_forearm_z      228.56280
```

Next, we run this model against our testing subset to estimate its
accuracy.


```r
rfPredict <- predict(rfModel, subTesting);
cmRF <- confusionMatrix(rfPredict, subTesting$classe)
cmRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    6    0    0    0
##          B    0  939   10    0    0
##          C    0    4  842    8    0
##          D    0    0    3  796    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9937         
##                  95% CI : (0.991, 0.9957)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.992          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9895   0.9848   0.9900   1.0000
## Specificity            0.9983   0.9975   0.9970   0.9993   1.0000
## Pos Pred Value         0.9957   0.9895   0.9859   0.9962   1.0000
## Neg Pred Value         1.0000   0.9975   0.9968   0.9981   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1915   0.1717   0.1623   0.1837
## Detection Prevalence   0.2857   0.1935   0.1741   0.1629   0.1837
## Balanced Accuracy      0.9991   0.9935   0.9909   0.9947   1.0000
```

Based on the above, we estimate this model has an accuracy of 
0.994 and an out of sample error of 1 - accuracy =
0.632 percent.

## Test Case Model Performance ##

Since the random forest model proved to be the most accurate, we
select that model and use it to make predictions against the test
dataset.


```r
rfTestPredict <- predict(rfModel, testing);
rfTestPredict
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The following code is used to submit our prediction results:


```r
for(i in 1:length(rfTestPredict)) {
   filename <- sprintf( "PMLtest%d", i);
   write.table(rfTestPredict[i], filename, append=FALSE, quote=FALSE, row.names=FALSE, col.names=FALSE);
}
```

The random forest model predicted all 20 of the 20 test cases correctly.

## Citations ##

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks,
H. Qualitative Activity Recognition of Weight Lifting
Exercises. Proceedings of 4th International Conference in Cooperation
with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI,
2013.
