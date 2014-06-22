### Project links:
```
Github Page For Project: https://github.com/KartikPadmanabhan/mlcoursera

gh-page branch for Project Report:  http://kartikpadmanabhan.github.io/mlcoursera/activity_predict.html

gh-page branch for Project Readme: http://kartikpadmanabhan.github.io/mlcoursera/ 
```

# Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 


## Data 

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

## Project Goals

The goal of our project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. We also created a report describing how we built our model, how we used cross validation, and what we think the expected out of sample error is, and why we made the choices we did. We also use our prediction model to predict 20 different test cases. 


## Dependencies

This project depends on the following packages installed through CRAN:
```
a. caret (for creating data partitions)
b. rpart (for tree modelling)
c. rpart.plot (for plotting better trees than plot() functions)
d. ipred (for bootstrap aggregation)
e. randomForest (for random forests)
f. knitrBootstrap (pls see below for bootstrap details)
```

## HTML Rendering

We make use of bootstrap to add beauty to html report. To install bootstrap please install it the following way:

```
library(devtools)
install_github('rstudio/rmarkdown')
install.packages('knitr', repos = c('http://rforge.net', 'http://cran.rstudio.org'),
                 type = 'source')
install_github('jimhester/knitrBootstrap')
```


