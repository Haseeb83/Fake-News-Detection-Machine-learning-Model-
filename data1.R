# Load libraries
library(dplyr)
library(ggplot2)
library(caret)
library(e1071)  # For Naive Bayes
library(corrplot)
library(Metrics)  # For Linear Regression metrics

# Load dataset
data1 <- read.csv("True.csv", stringsAsFactors = FALSE)
data2 <- read.csv("Fake.csv", stringsAsFactors = FALSE)

# Data Cleaning
data1 <- distinct(data1)  # Remove duplicates
data1[is.na(data1)] <- 0  # Replace NA with 0

# Ensure column types
data1$title <- as.character(data1$title)
data1$subject <- as.character(data1$subject)
data1$text <- as.character(data1$text)

# Ensure the `date` column is numeric
data1$date <- suppressWarnings(as.numeric(data1$date))
if (any(is.na(data1$date))) {
  data1$date[is.na(data1$date)] <- 0  # Replace invalid numeric conversions with 0
}

# Rename target column (if exists) or create it
if ("label" %in% colnames(data1)) {
  colnames(data1)[colnames(data1) == "label"] <- "target"
} else {
  # Assuming `target` is derived from another column, you need to define it
  data1$target <- sample(0:1, nrow(data1), replace = TRUE)  # Replace this with actual logic
}
data1$target <- as.factor(data1$target)

# Data Visualizations
numeric_cols <- data1[, sapply(data1, is.numeric), drop = FALSE]

# Histogram for numeric features
if (ncol(numeric_cols) > 0) {
  ggplot(stack(numeric_cols), aes(x = values)) +
    geom_histogram(fill = "blue", bins = 30, color = "black", alpha = 0.7) +
    facet_wrap(~ind, scales = "free") +
    ggtitle("Distribution of Numeric Features") +
    xlab("Values") +
    ylab("Frequency") +
    theme_minimal()
} else {
  cat("No numeric columns found for histogram.\n")
}

# Bar chart for class distribution
ggplot(data1, aes(x = target)) +
  geom_bar(fill = "orange") +
  ggtitle("Class Distribution: Fake vs Real News") +
  xlab("Target (0 = Fake, 1 = Real)") +
  ylab("Count") +
  theme_minimal()

# Correlation Analysis
if (ncol(numeric_cols) > 1) {
  correlation1 <- cor(numeric_cols, use = "complete.obs")
  corrplot(correlation1, method = "circle")
} else {
  cat("Insufficient numeric columns for correlation analysis.\n")
}

# Feature Selection
selected_features1 <- c(names(numeric_cols), "target")
data1 <- data1[, selected_features1, drop = FALSE]

# Split Data
set.seed(123)
train_index1 <- createDataPartition(data1$target, p = 0.8, list = FALSE)
train_data1 <- data1[train_index1, ]
test_data1 <- data1[-train_index1, ]

# Ensure target is numeric for linear regression
data1$target <- as.numeric(data1$target)

# Tuning grid for Naive Bayes
nb_tune_grid <- expand.grid(fL = 0, usekernel = TRUE, adjust = 1) 

# Train Naive Bayes model with hyperparameter tuning
nb_tune_model <- train(
  target ~ ., 
  data = train_data1, 
  method = "nb", 
  tuneGrid = nb_tune_grid, 
  trControl = trainControl(method = "cv", number = 5)  
)

# Predictions from Naive Bayes model
nb_pred1 <- predict(nb_tune_model, newdata = test_data1)

# Ensure target is numeric for regression
data1$target <- as.numeric(data1$target)


# Remove any rows with missing values
train_data1 <- na.omit(train_data1)

# Narrow tuning grid for glmnet (Elastic Net regularization)
lr_tune_grid <- expand.grid(
  alpha = c(0, 0.5, 1),  # Ridge, Elastic Net, and Lasso
  lambda = c(0, 0.01)  # Narrow lambda range
)

# Train the Linear Regression model using caret
train_control <- trainControl(
  method = "cv", 
  number = 5,  # Cross-validation with 5 folds
  summaryFunction = defaultSummary  # Use RMSE, MAE, or R-squared for regression
)

# Train the model
lr_tune_model <- train(
  target ~ ., 
  data = train_data1, 
  method = "glmnet", 
  tuneGrid = lr_tune_grid, 
  trControl = train_control
)

# Check if training was successful before printing the best tune parameters
if (exists("lr_tune_model")) {
  print(lr_tune_model$bestTune)
}

# Check for any warnings after the training
warnings()  # Use this to identify the reason for failure

# Predictions and evaluation if the model is trained successfully
if (exists("lr_tune_model")) {
  # Predictions
  lr_pred1 <- predict(lr_tune_model, newdata = test_data1)
  
  # RMSE and R-squared
  rmse_value <- sqrt(mean((lr_pred1 - test_data1$target)^2))  # RMSE calculation
  rsq_value <- cor(lr_pred1, test_data1$target)^2  # R-squared calculation
  
  cat(sprintf("RMSE: %.4f\n", rmse_value))
  cat(sprintf("R-squared: %.4f\n", rsq_value))
}

# Results Summary: Fake vs Real News
fake_news_count <- sum(test_data1$target == 0)
real_news_count <- sum(test_data1$target == 1)
cat(sprintf("Number of Fake News: %d\n", fake_news_count))
cat(sprintf("Number of Real News: %d\n", real_news_count))


