# Modeled off of "Extreme Gradient Boosting (XGBoost) with R and Python"
# Source: https://datascience-enthusiast.com/R/ML_python_R_part2.html

if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos='http://cran.rstudio.com/')
if(!require(caret)) install.packages("caret", repos='http://cran.rstudio.com/')
#if(!require(InformationValue)) install.packages("InformationValue", repos='http://cran.rstudio.com/')
library("readxl")
library("xgboost")
library("caret")
#library("InformationValue")

# Load and partition data.
dataset <- as.data.frame(read_excel("../../source/SkinSegmentation/SkinSegmentation.xlsx"))
partition <- createDataPartition(y = dataset$C, p = 0.75, list = FALSE)
train_data <- dataset[partition,]
test_data <- dataset[-partition,]

# Convert data into DMatrix format.
x_train = as.data.frame(train_data[1:3])
y_train = train_data$C
x_test = as.data.frame(test_data[1:3])
y_test = test_data$C

model <- glm(as.factor(C) ~ ., family=binomial(link='logit'), data=train_data)

# Calculate accuracy.
fitted.results <- predict(model,newdata=test_data,type='response')
fitted.results <- ifelse(fitted.results > 0.5,2,1)
error <- mean(fitted.results != test_data$C)
print(paste('Accuracy',1 - error))
summ <- summary(model)
print(summ)
