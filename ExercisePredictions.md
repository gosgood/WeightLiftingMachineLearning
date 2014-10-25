Prediction Exercise Quality
===========================
## Executive Summary

This analysis uses machine learning techniques to predict how "well" a user is performing weight lifting exercises.  We use a data set available from http://groupware.les.inf.puc-rio.br/har to serve as our training data.  The data set consists of data from personal fitness tracking devices that were worn while people lifted weights.

The analysis involves a high level exploration of the data, a simple method for selecting input features, the application of several machine learning algorithms, and analysis of results.  After looking at a few machine learning algorithms we chose Random Forest.  With Random Forest we were able to achive 99% accuracy on our training set, 99% accuracy on our hold-out test set, and 20/20 correct on the graded test cases.
    
## Exploratory Analysis

Let's take a high level look at the data set.




```r
data <- read.csv("pml-training.csv")
dim(data)
```

```
## [1] 19622   160
```

We see that the data set has 19622 observations with 160 variables, so this is a fairly large data set with many variables.  The majority of the variables represent location and movement at various positions (belt, arm, and dumbbell) during exercise.  With so many variables, it would be unwieldy to display a summary of all of them here, but we can quickly look at the distribution of the classe (outcome) variable:


```r
summary(data$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

There are a large number of samples for each possible outcome, which helps to create an accurate prediction algorithm.  

## Input Feature Selection

Displaying sample values (or even summaries) of all 160 variables in the data set would be unwieldy, but looking at the data in a spreadsheet reveals the following:

* Column 1 is simply an ID and isn't helpful for prediction.
* Column 2 is the user's name and won't generalize to new observations (with new users), and would not make a good input feature.
* Columns 3-5 are timestamps and also not likely to help for prediction.
* Columns 12-36, 50-59, 69-83, 87-101,103-112, 125-139, and 141-150 are mostly NAs or blank values and are unlikely to help with predictions.

Therefore, we will subset out the columns listed above and only use the remaining variables as input features.  The remaining variables are primarily numeric quantities.  We then partition our data set into training and testing sets using an 80/20 split, setting a seed beforehand to make our results reproducible.


```r
set.seed(123)
inputFeatures <- c(7:11,37:49,60:68,84:86,102,113:124,140,151:160)
dataClean <- data[,inputFeatures]
inTrain <- createDataPartition(y=dataClean$classe, p=.8, list=FALSE)
training <- dataClean[inTrain,]
testing <- dataClean[-inTrain,]
```

## Machine Learning

We tried several different machine learning algorithms on the data set, including classification trees, random forests, logistic regression, and even support vector machines.  In the end, random forest provided the best balance of computation time and accuracy.  Training with the randomForest function took about 1 minute, and resulted in a 99+% in-sample (training set) accuracy for each class and overall.  The code below shows the initial model with a confusion matrix for the training data:


```r
m1 <- randomForest(classe ~ ., data=training)
m1
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.22%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4463    1    0    0    0    0.000224
## B    3 3034    1    0    0    0.001317
## C    0   11 2726    1    0    0.004383
## D    0    0   13 2559    1    0.005441
## E    0    0    0    4 2882    0.001386
```

Note that the class.error column contains values below 1% for each class.  The following code shows the prediction results on the hold-out test set:


```r
predictions <- predict(m1, newdata=testing)
confusionMatrix(predictions,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    2    0    0    0
##          B    0  757    2    0    0
##          C    0    0  682    1    0
##          D    0    0    0  642    1
##          E    0    0    0    0  720
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.997, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.998         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.997    0.997    0.998    0.999
## Specificity             0.999    0.999    1.000    1.000    1.000
## Pos Pred Value          0.998    0.997    0.999    0.998    1.000
## Neg Pred Value          1.000    0.999    0.999    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.285    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    0.998    0.998    0.999    0.999
```

Again, the per-class and overall accuracy is 99+%, which gives us confidence that the model generalizes well.

Further evidence of the out-of-sample accuracy of random forest can be shown using the rfcv function, which uses cross-validation to measure accuracy of random forests.  By running rfcv with its default value of 5 folds, we get the following result:


```r
m2 <- rfcv(trainx=training[,-54], trainy=training[,54])
m2$error.cv
```

```
##        53        26        13         7         3         1 
## 0.0028027 0.0028664 0.0019746 0.0026753 0.0117205 0.0001274
```

Here we can see the error rates using cross-validation with 5 folds and varying numbers of predictors.  Even using only 7 predictors we see that the cross-validation error rate is already less than 1%, which gives us another good estimate of our out-of-sample error.
