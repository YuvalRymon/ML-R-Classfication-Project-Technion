# Fashion MNIST Image Classification Project
## 1. Dataset
- 60,000 28x28 grayscale images (784 pixels per image)
- 10 balanced fashion classes (~6000 each): 0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot

![image](https://github.com/user-attachments/assets/2ad9cbd3-c2c5-49c8-b5a8-f0031636bbf9)

 
## 2. Mission
Develop an accurate classifier for the 10 fashion categories, with particular focus on addressing difficult class distinctions.
## 3. Exploratory Data Analysis
### Data Cleaning
- Split: 70% training (42,000), 30% validation (18,000)
- Normalized pixel values to [0,1]
- Standardized features (μ=0, σ=1)
- Verified no missing values
### Summary Statistics
- Balanced class distribution (~4,200 training samples per class)
## 4. Principal Component Analysis (PCA)
**Purpose**: Reduce 784-dimensional feature space while preserving key information.
### Methodology
- Transforms correlated features into uncorrelated principal components
- Orders components by variance explained
- Enables dimensionality reduction with minimal information loss
### Results
- 80 components: 85% variance
- 136 components: 90% variance
- 255 components: 95% variance
- First two components plot reveals partial class separation with overlapping cluster, suggesting a need for non-linear classification

![image](https://github.com/user-attachments/assets/314c8193-aec5-41c8-bf7a-9189760c66b0)
![image](https://github.com/user-attachments/assets/6769ee68-a785-494b-b3b4-d70acac7bfb3)

  
## 5. Model Selection
Based on PCA visualization findings:
### QDA (Base Model)
Selected due to:
- Handles non-linear class boundaries
- Sufficient data volume mitigates variance concerns
- Works well with reduced dimensionality from PCA
### Random Forest (Advanced Model)
Selected due to:
- Naturally handles high-dimensional data
- Captures complex non-linear patterns
- Built-in feature importance analysis
- No need for PCA preprocessing
## 6. QDA Implementation
### Cross-validation
- 5-fold CV across PCA components (80, 136, 255)
- Best result: 80 components (70.05% accuracy)
- Performance decreases with more components
### Final Results
- Validation accuracy: 69.72%
- Notable confusion in class 6

![image](https://github.com/user-attachments/assets/fd7e7a44-c28c-4fe1-ad8b-8639f5be9cca)
 

## 7. Random Forest Implementation
### Hyperparameter Selection
1. ntree = 500
  - Balances accuracy and computation
  - Sufficient for stable predictions
2. nodesize = 1
  - Allows maximum tree growth
  - Captures fine-grained patterns
3. mtry optimization
  - Tests: 28 (sqrt(p)), 139 (p/6), 167 (p/5), 209 (p/4)
  - CV shows best value: 167

![image](https://github.com/user-attachments/assets/d31c3c66-1991-4de0-b814-b670ea9cf186)


### Feature Engineering
Addressing poor shirt (class 6) classification:
1. Enhanced Collar Feature
  - Combines intensity differences with edge detection
  - Captures collar area characteristics
2. Sleeve Shape Feature
  - Analyzes side edge patterns
  - Distinguishes sleeve types
3. Symmetry/Button Line Feature
  - Measures image symmetry
  - Detects central vertical patterns
4. Neckline Shape Feature
  - Edge detection in upper region
  - Classifies neckline types
5. Global Shape Descriptor
  - Implements Hu moments
  - Captures overall shape patterns

Features duplicated 10x to represent ~7% of total features
### Results
- Overall accuracy: 87.96%
- Shirt class improvements:
 - Precision: 73.26%
 - Recall: 60.00%
 - F1: 65.97
- New features rank highly
- ROC Curves: AUC > 0.95 for all classes

![image](https://github.com/user-attachments/assets/abc15e2c-c288-4a90-97cf-3a5776227cff)
![image](https://github.com/user-attachments/assets/acfc1577-9d19-4ba6-9a22-ccecda5a7067)
![image](https://github.com/user-attachments/assets/12273895-2181-497f-8f11-b49133db0be5)

 
## 8. Summary
Best in course performance achieved:
- 87.96% overall accuracy
- Significant improvement in shirt classification
- Strong AUC scores across all classes
- Custom features proven effective through importance rankings
The combination of Random Forest with targeted feature engineering successfully addressed the complex fashion classification task.
