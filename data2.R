# Load dataset
data1 <- read.csv("WELFake_Dataset.csv", stringsAsFactors = FALSE)

# Select the first 500 rows
data1 <- data1[1:500, ]

# Load libraries
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(corrplot)

# Data Cleaning
data1 <- distinct(data1)  # Remove duplicates
data1[is.na(data1)] <- 0  # Replace NA with 0
colnames(data1)[colnames(data1) == "label"] <- "target"
data1$target <- as.factor(data1$target)  # Convert target to factor

# Data Visualizations
# 1. Histogram for numeric features
numeric_cols <- data1[, sapply(data1, is.numeric), drop = FALSE]
ggplot(stack(numeric_cols), aes(x = values)) +
  geom_histogram(fill = "blue", bins = 30, color = "black", alpha = 0.7) +
  facet_wrap(~ind, scales = "free") +
  ggtitle("Distribution of Numeric Features") +
  xlab("Values") +
  ylab("Frequency") +
  theme_minimal()

# 2. Bar chart for class distribution
ggplot(data1, aes(x = target)) +
  geom_bar(fill = "orange") +
  ggtitle("Class Distribution: Fake vs Real News") +
  xlab("Target (0 = Fake, 1 = Real)") +
  ylab("Count") +
  theme_minimal()

# Line chart for trends (using numeric column averages over rows)
numeric_means <- colMeans(numeric_cols, na.rm = TRUE)
numeric_means_df <- data.frame(Feature = names(numeric_means), MeanValue = numeric_means)

ggplot(numeric_means_df, aes(x = Feature, y = MeanValue)) +
  geom_line(aes(group = 1), color = "green") +  # Add group aesthetic to ensure a single line
  geom_point(color = "red", size = 2) +
  ggtitle("Mean Value of Numeric Features") +
  xlab("Features") +
  ylab("Mean Value") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Correlation Analysis
correlation1 <- cor(numeric_cols)
corrplot(correlation1, method = "circle")

# Feature Selection
selected_features1 <- c(names(numeric_cols), "target")
data1 <- data1[, selected_features1, drop = FALSE]

# Split Data
set.seed(200)
train_index1 <- createDataPartition(data1$target, p = 0.8, list = FALSE)
train_data1 <- data1[train_index1, ]
test_data1 <- data1[-train_index1, ]

# --------------------------------------------
# Random Forest Hyperparameter Tuning
# Define the tuning grid for Random Forest
rf_grid <- expand.grid(
  mtry = c(1:5)           # Number of variables to try for splitting a node
)

# Train the Random Forest model using caret
rf_tune_model <- train(
  target ~ ., 
  data = train_data1, 
  method = "rf", 
  tuneGrid = rf_grid, 
  trControl = trainControl(method = "cv", number = 5)  # Cross-validation with 5 folds
)

# Print the best tuning parameters
print(rf_tune_model$bestTune)

# Get the predictions from the best model
rf_pred1_tuned <- predict(rf_tune_model, newdata = test_data1)

# Evaluate the model
rf_cm_tuned <- confusionMatrix(as.factor(rf_pred1_tuned), test_data1$target)
cat("Random Forest Confusion Matrix (Tuned):\n")
print(rf_cm_tuned)


# --------------------------------------------#
  # Support Vector Machine (SVM) Hyperparameter Tuning
  # Define the tuning grid for SVM with only C and sigma
  svm_grid <- expand.grid(
    C = c(0.1, 1, 10),       # Regularization parameter
    sigma = c(0.01, 0.1, 1)  # Sigma values for the radial basis function
  )
  
  # Train the SVM model using caret
  svm_tune_model <- tryCatch({
    train(
      target ~ ., 
      data = train_data1, 
      method = "svmRadial",  # Radial kernel for SVM
      tuneGrid = svm_grid, 
      trControl = trainControl(method = "cv", number = 5)  # Cross-validation with 5 folds
    )
  }, error = function(e) {
    cat("Error in training SVM model: ", e$message, "\n")
    NULL  # Return NULL in case of error
  })
  
  # Check if the model was trained successfully before printing the best tune parameters
  if (!is.null(svm_tune_model)) {
    # Print the best tuning parameters
    print(svm_tune_model$bestTune)
  } else {
    cat("SVM model training failed, no best tune parameters to print.\n")
  }
  


# Get the predictions from the best model
svm_pred1_tuned <- predict(svm_tune_model, newdata = test_data1)

# Evaluate the model
svm_cm_tuned <- confusionMatrix(as.factor(svm_pred1_tuned), test_data1$target)
cat("SVM Confusion Matrix (Tuned):\n")
print(svm_cm_tuned)

# --------------------------------------------
# Results Summary: Fake vs Real News
fake_news_count <- sum(test_data1$target == 0)
real_news_count <- sum(test_data1$target == 1)
cat(sprintf("Number of Fake News: %d\n", fake_news_count))
cat(sprintf("Number of Real News: %d\n", real_news_count))

