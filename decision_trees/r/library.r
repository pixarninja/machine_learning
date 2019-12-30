# Modeled off of "Extreme Gradient Boosting (XGBoost) with R and Python"
# Source: https://datascience-enthusiast.com/R/ML_python_R_part2.html

if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos='http://cran.rstudio.com/')
if(!require(caret)) install.packages("caret", repos='http://cran.rstudio.com/')
library("readxl")
library("xgboost")
library("caret")

# Load and partition data.
dataset <- as.data.frame(read_excel("../../source/CCPP/Folds5x2_pp.xlsx"))
partition <- createDataPartition(y = dataset$PE, p = 0.75, list = FALSE)
train_data <- dataset[partition,]
test_data <- dataset[-partition,]

# Convert data into DMatrix format.
x_train = xgb.DMatrix(as.matrix(train_data[1:4]))
y_train = train_data$PE
x_test = xgb.DMatrix(as.matrix(test_data[1:4]))
y_test = test_data$PE

xgb_model = train(
  x_train, y_train,  
  method = "xgbTree"
)

# Calculate Mean Squared Error.
predicted = predict(xgb_model, x_test)
loss = (y_test - predicted)^2
error = sqrt(mean(loss))
cat('Mean Squared Error (MSE): ', error,'\n')

# Calculate R-squared.
y_test_mean = mean(y_test)
tss =  sum((y_test - y_test_mean)^2 )

residuals = y_test - predicted
rss =  sum(residuals^2)

r  =  1 - (rss/tss)
cat('R-squared: ', r, '\n')

# Write to file for plotting in Python.
losses <- c()
for (i in seq(1, by=1, length=length(y_test)))
	losses[i] = (y_test[i] - predicted[i])^2;

write.table(losses, file = "output_losses.txt", sep = "\n",
            row.names = FALSE)
write.table(y_test, file = "output_y_test.txt", sep = "\n",
            row.names = FALSE)
