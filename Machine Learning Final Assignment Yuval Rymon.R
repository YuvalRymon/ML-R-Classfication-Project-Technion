# Load necessary libraries
library(tidyverse)               # For data manipulation and visualization
library(ggplot2)                 # For plotting
library(reshape2)                # For reshaping data
library(dplyr)                   # For data manipulation
library(caret)                   # For machine learning
library(randomForest)            # For Random Forest
library(doParallel)              # For parallel processing
library(pROC)
library(viridis)
library(MASS)
library(parallel)
# Set working directory to your project folder
setwd("C:\\Yuval\\Studying\\Study in Israel\\Technion Data Science\\קורס\\עבודות להגשה\\Assignment 2 - Statistical Learning")

# ------------------------- Data Cleaning and Preparation ------------------------------------

# Load the Fashion MNIST dataset
mnist_train <- read_csv("fashion-mnist_train.csv")

# Separate features and labels using dplyr::select to avoid masking issues
X <- mnist_train %>% dplyr::select(-label)  # Features
y <- mnist_train %>% dplyr::select(label)    # Labels

# Generate random indices for splitting
set.seed(123)
train_ratio <- 0.70 ; val_ratio <- 0.30
n <- nrow(X)
train_indices <- sample(1:n, size = floor(train_ratio * n))
remaining_indices <- setdiff(1:n, train_indices)
val_indices <- sample(remaining_indices, size = floor(val_ratio * n))

# Split the data and normalize pixel values
X_train <- X[train_indices, ] ; y_train <- y[train_indices, ]
X_val <- X[val_indices, ] ; y_val <- y[val_indices, ]
X_train <- X_train / 255 ; X_val <- X_val / 255

# Summary of lables and features
summary(y_train)  # Summary of labels
summary(X_train[, 1:10])  # Check first 10 features

# Check for missing values
sum(is.na(X_train)) ; sum(is.na(y_train)) 
sum(is.na(X_val)) ; sum(is.na(y_val))

# Visualize the distribution of classes (labels)
ggplot(y_train, aes(x = factor(label), fill = factor(label))) +
  geom_bar(show.legend = FALSE, width = 0.6) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Distribution of Fashion Classes",
    x = "Class Label",
    y = "Count"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title.x = element_text(size = 12, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Standardize the training data
X_train_scaled <- scale(X_train)
# Calculate mean and standard deviation from the training data
mean_X_train <- attr(X_train_scaled, "scaled:center")
sd_X_train <- attr(X_train_scaled, "scaled:scale")

# Standardize the validation data using the same approach
X_val_scaled <- scale(X_val, center = mean_X_train, scale = sd_X_train)

# Convert y_train and y_val to a factor (for classification)
y_train$label <- as.factor(y_train$label)
y_val <- data.frame(label = as.factor(y_val$label))

# ------------------------- PCA IMPLEMENTATION ------------------------------------

# Perform PCA to reduce dimensionality
pca_result <- prcomp(X_train_scaled, center = TRUE, scale. = TRUE)

summary(pca_result)

# Display PCA summary to check structure
pca_summary <- summary(pca_result)
print(pca_summary)

# Cumulative variance explained by principal components
cumulative_variance <- pca_summary$importance[3, ]
explained_variance <- pca_summary$importance[2, ]  # This is for individual proportions of variance

# Print explained variance and cumulative variance
print(explained_variance)
print(cumulative_variance)

# Plot commulative variance and lines for 85%, 90% and 95%
plot(cumulative_variance, type = "b", xlab = "Number of Principal Components", 
      ylab = "Cumulative Proportion of Variance Explained", 
      main = "PCA: Cumulative Variance Explained")
 abline(h = 0.95, col = "red", lty = 2)  # Line for 95% variance
 abline(h = 0.90, col = "red", lty = 2)  # Line for 90% variance
 abline(h = 0.85, col = "red", lty = 2)  # Line for 85% variance


# Find the number of components needed to explain 85%, 90%, and 95% of the variance
num_components_85 <- which(cumulative_variance >= 0.85)[1]
num_components_90 <- which(cumulative_variance >= 0.90)[1]
num_components_95 <- which(cumulative_variance >= 0.95)[1]
cat("Number of components needed for 85% variance:", num_components_85, "\n")
cat("Number of components needed for 90% variance:", num_components_90, "\n")
cat("Number of components needed for 95% variance:", num_components_95, "\n")
# 80 components explain 85% of the variance, 136 components explain 90%, and 255 components explain 95%.
# For a balance between dimensionality reduction and preserving sufficient information, I will be using 136
# principal components, which capture 90% of the variance while significantly reducing the dimensionality from 784 to 136.

# Visualizing Principal Components to see how well the different classes are separated in lower dimensions.
# If the classes (labels) are well-separated in the first two or three principal components,
# it suggests that simpler models like LDA may work well, as these assume that the data is somewhat linearly separable. 
pca_df <- data.frame(PC1 = pca_result$x[, 1], PC2 = pca_result$x[, 2], label = y_train$label)
ggplot(pca_df, aes(x = PC1, y = PC2, color = as.factor(label))) +
   geom_point(alpha = 0.6) +
   labs(title = "PCA: First Two Principal Components", x = "Principal Component 1", y = "Principal Component 2") +
   theme_minimal()

# The plot shows that there are clusters of points that seem to be grouped by class (like Class 0 and Class 1).
# However, there is also some overlap between certain classes, especially in the central region of the plot.
# This suggests that while some classes are reasonably separated, others might have a more complex relationship that could lead to misclassifications. 
# Overall, the visualization indicates that the dataset may not be entirely linearly separable. 

# ------------------------- QDA IMPLEMENTATION ------------------------------------

# In light of the PCA, I will try QDA as my first benchmark model. While LDA could work reasonably well
# for classes that are more distinctly separated, QDA might provide better results for the overlapping classes.
# In addition, due to the large amount of observations, the variance is less of an issue.

# Set seed for reproducibility
set.seed(123)
 
# Define the number of PCs to try
pc_options <- c(80, 136, 255)
 
# Create a custom control object for cross-validation
ctrl <- trainControl(
   method = "cv",  # Use k-fold cross-validation
   number = 5,     # Number of folds
   classProbs = TRUE,  # Compute class probabilities
   allowParallel = TRUE  # Allow parallel processing
)

# Function to convert numeric labels to factor with valid R variable names
convert_labels <- function(y) {
  factor(y, levels = 0:9, labels = paste0("Class_", 0:9))
}
# Function to train and evaluate QDA model for a given number of PCs
train_qda <- function(n_pc, X, y) {
  # Extract the specified number of PCs
  X_pca <- pca_result$x[, 1:n_pc]
  
  # Convert labels to factor with valid R variable names
  y_factor <- convert_labels(y)
  
  # Combine PCA features with labels
  train_data <- data.frame(X_pca, label = y_factor)
  
  # Train QDA model using cross-validation
  model <- train(
    label ~ .,
    data = train_data,
    method = "qda",
    trControl = ctrl
  )
  
  return(model)
}

# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Train models for each PC option
models <- lapply(pc_options, function(n_pc) train_qda(n_pc, X_train_scaled, y_train$label))

# Stop parallel processing
stopCluster(cl)

# Extract accuracies
accuracies <- sapply(models, function(model) max(model$results$Accuracy))

# Find the best number of PCs
best_pc <- pc_options[which.max(accuracies)]
best_model <- models[[which.max(accuracies)]]

# Print results
cat("Cross-validation results:\n")
for (i in seq_along(pc_options)) {
  cat(sprintf("%d PCs: Accuracy = %.4f\n", pc_options[i], accuracies[i]))
}
cat(sprintf("\nBest number of PCs: %d\n", best_pc))

# Use the best model to make predictions on the validation set
X_val_pca <- predict(pca_result, X_val_scaled)[, 1:best_pc]
y_val_factor <- convert_labels(y_val$label)
val_predictions <- predict(best_model, newdata = data.frame(X_val_pca))

# Calculate accuracy on validation set
val_accuracy <- mean(val_predictions == y_val_factor)
cat(sprintf("Validation Accuracy: %.4f\n", val_accuracy))

# Generate confusion matrix
conf_matrix <- confusionMatrix(val_predictions, y_val_factor)
print(conf_matrix)

# Plot confusion matrix
library(ggplot2)
conf_data <- as.data.frame(conf_matrix$table)
ggplot(conf_data, aes(Prediction, Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal() +
  labs(title = "Confusion Matrix", x = "Predicted", y = "Actual")


# ------------------------- RANDOM FOREST IMPLEMENTATION --------------------------

# Now, I will deploy a Random Forest which may work better then the QDA due to its ability to handle large feature spaces
# (and also nonlinear relationships like QDA). While PCA is effective for reducing dimensionality and removing noise for the QDA,
# Random Forest can inherently manage high-dimensional data. It constructs many decision trees based on the bootstrap of a sub sample
# of the data and random subsets of features, making it robust against irrelevant features. # Thus, applying PCA before using RF might
# result in the loss of potentially informative data, and therefore I will not use it.

# Hyperparameters chosen for Random Forest:
# 1. ntree: I chose 500 trees, as more trees can improve model accuracy and I aimed to reach 90% accuracy.
# 2. mtry: I chose to do CV with the default (sqrt num features = 28), and 10%, 20%, 33% of features (81,162,271)
# This HP is important for complexity and generally should be large in image classification.
# 3. sampsize: I chose 70% of the training data to be used for the bootstrap.
# Using a smaller sample size can help the model generalize better.
# 4. nodesize: I saw that in largest forests 1 was optimal and therefore chose it as the minimum size of terminal nodes in each tree.
# Lower sizes allow each tree to be more flexible, which can help with capturing more detail in the data.

# In light of the specific accuracy problem in classifying class 6 (shirt), I added 5 new features designed to capture its characteristics:
# 1. Enhanced Collar Feature: Combines collar intensity difference with edge detection to capture both intensity and shape of the collar area.
# 2. Sleeve Shape Feature: Measures edge characteristics on the sides of the upper image to differentiate short-sleeved from long-sleeved items.
# 3. Symmetry and Button Line Feature: Combines overall image symmetry with central vertical line asymmetry to identify button-up shirts.
# 4. Neckline Shape Feature: Uses edge detection and contour analysis on the top 20% of the image to classify different neckline types.
# 5. Global Shape Descriptor: Implements simplified Hu moments to capture overall shape characteristics for broad item categorization.
# I also multiplied them by 10 to make them ~7% of features and contribute to the model. 

create_shirt_features <- function(data) {
  if (!is.matrix(data)) {
    data <- as.matrix(data)
  }
  
  if (ncol(data) == 784) {
    data <- array(t(data), dim = c(28, 28, nrow(data)))
    data <- aperm(data, c(3, 1, 2))
  } else if (length(dim(data)) != 3) {
    stop("Input data must be either a 2D matrix with 784 columns or a 3D array")
  }
  
  n_samples <- dim(data)[1]
  
  enhanced_collar_feature <- numeric(n_samples)
  symmetry_button_feature <- numeric(n_samples)
  sleeve_shape_feature <- numeric(n_samples)
  neckline_shape_feature <- numeric(n_samples)
  global_shape_feature <- numeric(n_samples)
  
  # Simple edge detection function
  edge_detect <- function(img) {
    dx <- diff(img)
    dy <- diff(t(img))
    return(mean(abs(dx)) + mean(abs(dy)))
  }
  
  for (i in 1:n_samples) {
    img <- data[i,,]
    
    # Enhanced Collar Feature
    collar_area <- img[1:7, 10:20]
    surrounding_area <- img[1:7, c(1:9, 21:28)]
    intensity_diff <- mean(collar_area) - mean(surrounding_area)
    edge_intensity <- edge_detect(img[1:10,])
    enhanced_collar_feature[i] <- intensity_diff * edge_intensity
    
    # Symmetry and Button Line Feature
    left_half <- img[, 1:14]
    right_half <- img[, 15:28]
    symmetry_score <- cor(as.vector(left_half), as.vector(right_half))
    center_column <- img[, 14:15]
    button_line_score <- var(as.vector(center_column))
    symmetry_button_feature[i] <- symmetry_score * button_line_score
    
    # Sleeve Shape Feature
    left_sleeve <- img[8:20, 1:5]
    right_sleeve <- img[8:20, 24:28]
    sleeve_edges <- edge_detect(cbind(left_sleeve, right_sleeve))
    sleeve_shape_feature[i] <- sleeve_edges
    
    # Neckline Shape Feature
    neckline_area <- img[1:7, ]
    neckline_edges <- edge_detect(neckline_area)
    neckline_shape_feature[i] <- neckline_edges
    
    # Global Shape Descriptor (simplified Hu moments)
    m00 <- sum(img)
    m10 <- sum(row(img) * img)
    m01 <- sum(col(img) * img)
    xbar <- m10 / m00
    ybar <- m01 / m00
    mu20 <- sum((row(img) - xbar)^2 * img) / m00
    mu02 <- sum((col(img) - ybar)^2 * img) / m00
    global_shape_feature[i] <- mu20 + mu02  # First Hu moment
  }
  
  new_features <- data.frame(
    enhanced_collar_feature = enhanced_collar_feature,
    symmetry_button_feature = symmetry_button_feature,
    sleeve_shape_feature = sleeve_shape_feature,
    neckline_shape_feature = neckline_shape_feature,
    global_shape_feature = global_shape_feature
  )
  
  return(new_features)
}

# Function to scale new features
scale_new_features <- function(features, center = NULL, scale = NULL) {
  if (is.null(center) && is.null(scale)) {
    # For training data: calculate and return scaling parameters
    scaled_features <- scale(features)
    return(list(
      scaled = scaled_features,
      center = attr(scaled_features, "scaled:center"),
      scale = attr(scaled_features, "scaled:scale")
    ))
  } else {
    # For validation data: use provided scaling parameters
    return(scale(features, center = center, scale = scale))
  }
}

# Create and scale new features for training data
new_train_features <- create_shirt_features(X_train_scaled)
scaled_train_features <- scale_new_features(new_train_features)


# New feature multiplication for enhanced weight
emphasize_features <- function(X, feature_names, duplication_factor = 10) {
  for (feature in feature_names) {
    for (i in 1:duplication_factor) {
      new_name <- paste0(feature, "_copy", i)
      X[[new_name]] <- X[[feature]]
    }
  }
  return(X)
}

# Duplicate new features
new_feature_names <- c("enhanced_collar_feature", "symmetry_button_feature", "sleeve_shape_feature", 
                       "neckline_shape_feature", "global_shape_feature")
duplicated_train_features <- emphasize_features(as.data.frame(scaled_train_features$scaled), new_feature_names)

# Combine scaled original features with new scaled features
enhanced_train_data <- cbind(X_train_scaled, duplicated_train_features)

# Update training_data to include new scaled features
training_data <- data.frame(label = y_train$label, enhanced_train_data)

#can reuse if lets go of 6 improvement:
# Combine X_train_scaled and y_train for modeling
#training_data <- data.frame(label = y_train$label, X_train_scaled)
#str(training_data)

# Set up parallel processing
library(doParallel)
num_cores <- detectCores() - 1  # Leave one core free
cl <- makeCluster(num_cores)
registerDoParallel(cl)


# Function to perform k-fold cross-validation
rf_cv <- function(nodesize = 1, k = 5, ntree = 500, mtry) {
  set.seed(123)
  folds <- createFolds(training_data, k = k, list = TRUE, returnTrain = FALSE)
  cv_errors <- numeric(k)
  cat(sprintf("\nStarting cross-validation for mtry = %d\n", mtry))
  
  
  for(i in 1:k) {
    train <- training_data[-folds[[i]], ]
    test <- training_data[folds[[i]], ]
    
    rf_model <- randomForest(label ~ ., 
                             data = train, 
                             ntree = ntree, 
                             mtry = mtry,
                             nodesize = nodesize,
                             sampsize = floor(0.7 * nrow(train)),
                             parallel = TRUE)  # Enable parallel processing
    predictions <- predict(rf_model, newdata = test)
    cv_errors[i] <- mean(predictions != test$label)
    cat(sprintf(" Error: %.4f\n", cv_errors[i]))
    
  }
  mean_cv_error <- mean(cv_errors)
  cat(sprintf("Completed cross-validation for mtry = %d. Mean CV Error: %.4f\n", mtry, mean_cv_error))
  return(list(cv_error = mean(cv_errors), sd_error = sd(cv_errors)))
}

# Function to train and evaluate model with CV
train_evaluate_cv <- function(mtry) {
  cv_result <- rf_cv(mtry = mtry)
  return(list(mtry = mtry, 
              cv_error = cv_result$cv_error, 
              cv_error_sd = cv_result$sd_error))
}

# Export necessary objects and functions to the cluster
clusterEvalQ(cl, {
  library(randomForest)
  library(caret)
})
clusterExport(cl, c("rf_cv", "train_evaluate_cv", "training_data"))


# mtry values
num_features <- ncol(training_data) - 1
mtry_values <- c(
  floor(sqrt(num_features)),             # Default: sqrt(p)
  floor(num_features/6),
  floor(num_features/5),         
  floor(num_features/4)
  )
  
print(paste("Testing mtry values:", paste(mtry_values, collapse = ", ")))

# Start timing
start_time <- Sys.time()

# Perform CV for different mtry
results <- parLapply(cl, mtry_values, train_evaluate_cv)

# Stop timing
end_time <- Sys.time()
training_time <- end_time - start_time
print(paste("Training Time:", training_time))

# Stop the cluster after training
stopCluster(cl)

# Combine results into a data frame
results_df <- bind_rows(results)

# Print results
print(results_df)

# Identify best mtry
best_mtry <- results_df$mtry[which.min(results_df$cv_error)]
cat("Best mtry based on CV error:", best_mtry, "\n")

cv_error_plot <- ggplot(results_df, aes(x = mtry, y = cv_error)) +
  geom_line(color = "darkblue", size = 1) +
  geom_point(color = "darkblue", size = 4) +
  geom_errorbar(aes(ymin = cv_error - cv_error_sd, 
                    ymax = cv_error + cv_error_sd), 
                width = 0.1, color = "darkblue", alpha = 0.7) +
  labs(title = "Cross-Validation Error by mtry",
       subtitle = "Random Forest Model Tuning",
       x = "mtry",
       y = "Cross-Validation Error") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    plot.subtitle = element_text(hjust = 0.5, face = "italic", size = 12),
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(size = 10),
    panel.grid.minor = element_blank()
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  scale_x_continuous(breaks = unique(results_df$mtry)) +
  geom_text(aes(label = sprintf("%.2f%%", cv_error * 100)), 
            vjust = -1.5, size = 3.5, color = "darkblue") +
  coord_cartesian(ylim = c(min(results_df$cv_error - results_df$cv_error_sd) * 0.95, 
                           max(results_df$cv_error + results_df$cv_error_sd) * 1.05))

# Display the plot
print(cv_error_plot)
# Start timing
start_time <- Sys.time()
# Train final model with best nodesize
final_model <- randomForest(label ~ ., 
                            data = training_data,
                            ntree = 500,
                            mtry = best_mtry,
                            nodesize = 1,
                            sampsize = floor(0.7 * nrow(training_data)),
                            importance = TRUE,
                            parallel = TRUE)  # Enable parallel processing

# Stop timing
end_time <- Sys.time()
training_time <- end_time - start_time
print(paste("Training Time:", training_time))
# Print final model
print(final_model)

# Make predictions on validation set
new_val_features <- create_shirt_features(X_val_scaled)
scaled_val_features <- scale_new_features(new_val_features, 
                                          center = scaled_train_features$center, 
                                          scale = scaled_train_features$scale)

# Duplicate new features for validation data
duplicated_val_features <- emphasize_features(as.data.frame(scaled_val_features), new_feature_names)

X_val_enhanced <- cbind(X_val_scaled, duplicated_val_features)
validation_predictions <- predict(final_model, newdata = X_val_enhanced)

#can reuse if let go of 6 improvement:
# Make predictions on validation set
#X_val_scaled_df <- as.data.frame(X_val_scaled)
#validation_predictions <- predict(final_model, newdata = X_val_scaled_df)

# Calculate accuracy on validation set
validation_accuracy <- mean(validation_predictions == y_val$label)
cat("Validation Accuracy:", validation_accuracy, "\n")

# Create confusion matrix
conf_matrix <- table(Predicted = validation_predictions, Actual = y_val$label)
print(conf_matrix)

# Plot confusion matrix
conf_matrix_df <- as.data.frame(conf_matrix)
conf_matrix_df$Predicted <- as.factor(conf_matrix_df$Predicted)
conf_matrix_df$Actual <- as.factor(conf_matrix_df$Actual)

ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal() +
  labs(title = "Confusion Matrix",
       x = "Actual Class",
       y = "Predicted Class") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("confusion_matrix.png", width = 10, height = 8)

# Feature Importance
importance_scores <- importance(final_model)
importance_df <- data.frame(Feature = rownames(importance_scores), 
                            Importance = importance_scores[, "MeanDecreaseGini"])
importance_df <- importance_df %>% 
  arrange(desc(Importance)) %>% 
  mutate(Feature = factor(Feature, levels = Feature))

grouped_importance <- importance_df %>%
  mutate(Base_Feature = sub("_copy\\d+$", "", Feature)) %>%
  group_by(Base_Feature) %>%
  summarise(Total_Importance = sum(Importance)) %>%
  arrange(desc(Total_Importance))

print("Grouped importance of features (including duplicates):")
print(head(grouped_importance, 20))

# Select top 20 grouped features
top_20_grouped <- grouped_importance %>%
  top_n(20, Total_Importance) %>%
  mutate(Base_Feature = fct_reorder(Base_Feature, Total_Importance))

# Create an enhanced plot for grouped importance
importance_plot <- ggplot(top_20_grouped, aes(x = Base_Feature, y = Total_Importance, fill = Total_Importance)) +
  geom_bar(stat = "identity", width = 0.8) +
  coord_flip() +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Top 20 Grouped Feature Importance in Random Forest Model",
       subtitle = "Based on Mean Decrease in Gini Index (Including Duplicates)",
       x = "",
       y = "Total Importance Score") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    plot.subtitle = element_text(hjust = 0.5, face = "italic", size = 12),
    axis.title.x = element_text(face = "bold", size = 12),
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 10),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.x = element_blank(),
    legend.position = "none"
  ) +
  geom_text(aes(label = sprintf("%.3f", Total_Importance)), hjust = -0.1, size = 3.5)

# Display the plot
print(importance_plot)

# ROC Curve
# Get predicted probabilities for each class
pred_probs <- predict(final_model, newdata = X_val_enhanced, type = "prob")
roc_list <- lapply(1:10, function(i) {
  roc(y_val$label == levels(y_val$label)[i], pred_probs[,i])
})

# Plot ROC curves
roc_data <- lapply(1:10, function(i) {
  data.frame(
    tpr = roc_list[[i]]$sensitivities,
    fpr = 1 - roc_list[[i]]$specificities,
    class = levels(y_val$label)[i]
  )
})
roc_data <- do.call(rbind, roc_data)

ggplot(roc_data, aes(x = fpr, y = tpr, color = class)) +
  geom_line() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "ROC Curves for Each Class",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_minimal()

# Save the plot
dev.copy(png, "roc_curves.png", width = 800, height = 600)
dev.off()

# Calculate and print AUC for each class
auc_values <- sapply(roc_list, auc)
print(data.frame(Class = levels(y_val$label), AUC = auc_values))

#-------------------------tests for new features and class 6--------------------
# Check importance of new features
new_features_importance <- importance_df %>%
  filter(grepl("enhanced_collar_feature|symmetry_button_feature|sleeve_shape_feature|neckline_shape_feature|global_shape_feature", Feature))
print("Importance of new features:")
print(new_features_importance)

# Analyze feature importance for class 6 (shirts)
class_6_importance <- importance(final_model, class=6)
class_6_importance_df <- data.frame(Feature = rownames(class_6_importance),
                                    Importance = class_6_importance[, "MeanDecreaseAccuracy"]) %>%
  arrange(desc(Importance))

print("Top 20 features for Class 6 (Shirts):")
print(head(class_6_importance_df, 20))

# Check rank of new features for Class 6
new_features_rank_class_6 <- which(grepl("enhanced_collar_feature|symmetry_button_feature|sleeve_shape_feature|neckline_shape_feature|global_shape_feature", class_6_importance_df$Feature))
print("Rank of new features for Class 6:")
print(new_features_rank_class_6)

shirt_actual <- y_val$label == "6"
shirt_predicted <- validation_predictions == "6"
shirt_precision <- sum(shirt_predicted & shirt_actual) / sum(shirt_predicted)
shirt_recall <- sum(shirt_predicted & shirt_actual) / sum(shirt_actual)
shirt_f1 <- 2 * (shirt_precision * shirt_recall) / (shirt_precision + shirt_recall)

print(paste("Shirt precision:", shirt_precision))
print(paste("Shirt recall:", shirt_recall))
print(paste("Shirt F1 score:", shirt_f1))

#--------------------------------END--------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------