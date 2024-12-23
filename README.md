# Statistical Learning Final Assignment

This repository contains an R script for classifying the Fashion MNIST dataset. The project includes data preprocessing, feature engineering, dimensionality reduction, and training machine learning models like QDA and Random Forest. It also emphasizes model optimization and evaluation using visualization and statistical techniques.

## Project Structure

The R script is divided into the following segments:

### 1. **Setup and Libraries**
This segment loads the necessary libraries and sets the working directory.

**Key Functions:**
- Loads libraries for data manipulation (`tidyverse`, `dplyr`), visualization (`ggplot2`), machine learning (`caret`, `randomForest`), and evaluation (`pROC`).
- Sets the working directory to the folder containing the dataset.

### 2. **Data Loading and Preprocessing**
This segment handles the loading, cleaning, and preprocessing of the Fashion MNIST dataset.

**Key Steps:**
- Loads the Fashion MNIST dataset.
- Splits the data into training and validation sets (70% training, 30% validation).
- Normalizes pixel values for consistent scaling.
- Checks for missing values and visualizes class distributions.

### 3. **Dimensionality Reduction with PCA**
Principal Component Analysis (PCA) is used to reduce the dataset's dimensionality while retaining 90% of the variance.

**Key Outputs:**
- PCA summary and variance explained.
- Number of components required for 85%, 90%, and 95% variance.
- Visualization of the cumulative variance explained.

### 4. **Quadratic Discriminant Analysis (QDA)**
Implements QDA as an initial benchmark model. The dataset is transformed using PCA for dimensionality reduction before applying QDA.

**Key Features:**
- Performs cross-validation to evaluate accuracy for different PCA components.
- Identifies the optimal number of components based on cross-validation results.

### 5. **Random Forest Classifier**
Trains and tunes a Random Forest model, focusing on hyperparameter optimization and feature importance analysis.

**Key Steps:**
- Custom feature engineering to improve classification for challenging classes (e.g., shirts).
- Hyperparameter tuning for `mtry`, `ntree`, `sampsize`, and `nodesize` to optimize model accuracy.
- Visualization of cross-validation errors for different `mtry` values.

**Custom Features Added for Class 6 (Shirts):**
1. Enhanced Collar Feature.
2. Sleeve Shape Feature.
3. Symmetry and Button Line Feature.
4. Neckline Shape Feature.
5. Global Shape Descriptor.

### 6. **Model Evaluation**
Comprehensive evaluation of the trained models using validation data.

**Evaluation Metrics:**
- Confusion matrices.
- Precision, recall, and F1 scores for specific classes (e.g., shirts).
- Feature importance analysis.
- ROC curves and AUC scores for each class.

**Key Visualizations:**
- Confusion matrix heatmaps.
- Feature importance bar plots.
- ROC curves for all classes.

## Results Summary
- **PCA**: Reduced dimensions to 136 components while retaining 90% variance.
- **QDA**: Achieved reasonable accuracy but was outperformed by Random Forest due to nonlinear relationships in the data.
- **Random Forest**: Optimized model achieved high accuracy and significant improvements in classifying shirts.

## Libraries Used
- **Data Manipulation**: `tidyverse`, `dplyr`, `reshape2`
- **Visualization**: `ggplot2`, `viridis`
- **Machine Learning**: `caret`, `randomForest`
- **Evaluation**: `pROC`
- **Parallel Processing**: `doParallel`, `parallel`

## How to Use
1. Clone this repository to your local machine.
2. Place the Fashion MNIST dataset files (`fashion-mnist_train.csv`) in the working directory.
3. Update the `setwd()` path in the script to point to your dataset location.
4. Run the script step-by-step in R to preprocess data, train models, and evaluate their performance.

## Future Directions
- Experiment with additional models (e.g., SVM, Gradient Boosting).
- Extend feature engineering to improve classification of other challenging classes.
- Apply the methods to other datasets for broader evaluation.

## Acknowledgments
This project was completed as part of a Statistical Learning assignment. It demonstrates the application of machine learning and statistical techniques for practical classification problems.

---

Feel free to copy and adapt this README for your project!
