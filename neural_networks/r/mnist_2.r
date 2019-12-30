# Modeled off of "Keras for R Studio"
# Source: https://blog.rstudio.com/2017/09/05/keras-for-r/

if(!require(keras)) {
  install.packages("keras")
  install_keras()
}
library("keras")
setwd("./mnist/")

# Load dataset.
data <- dataset_mnist()
x_train <- data$train$x
y_train <- data$train$y
x_test <- data$test$x
y_test <- data$test$y

# Reshape data.
dim(x_train) <- c(nrow(x_train), 28, 28, 1)
dim(x_test) <- c(nrow(x_test), 28, 28, 1)
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Create model.
model <- keras_model_sequential() 
model %>% 
  layer_conv_2d(filter = 64, kernel_size = c(3,3), input_shape = c(28,28,1)) %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 64, kernel_size = c(3,3), input_shape = c(28,28,1)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  layer_flatten() %>% 
  layer_dense(units = 128, activation = "relu", input_shape = c(784)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 10, activation = "softmax")

# Compile model.
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# Obtain fit.
history <- model %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 128, 
  validation_split = 0.2
)

# Evaluate fit.
plot(history)
model %>% evaluate(x_test, y_test,verbose = 0)
