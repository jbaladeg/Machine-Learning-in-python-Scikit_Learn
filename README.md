# Scikit-learn Classification with Digits Dataset

## Introduction

This repository contains an educational Python script demonstrating how to use **Scikit-learn** for classification tasks using the **digits dataset**.  
It covers different algorithms and techniques:

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression** (including cross-validation and GridSearchCV)
- **Data preprocessing and normalization**
- **Evaluation metrics** including accuracy, confusion matrix, and classification reports

---

## 1. Data Preparation

The script uses the **digits dataset** from Scikit-learn:

- `digits.data`: NumPy array of shape `(1797, 64)`. Each row is an 8x8 image flattened into 64 features, representing grayscale pixel intensities (0=white, 16=black).  
- `digits.target`: Array of shape `(1797,)` with labels 0–9 indicating the digit.  
- `digits.feature_names` and `digits.target_names` provide descriptive names.

The data is split into **training and test sets (75%-25%)**. Features are normalized to have mean 0 and standard deviation 1 using `StandardScaler` to improve model performance.

---

## 2. K-Nearest Neighbors (KNN)

KNN predicts the class of a sample based on the majority class among its k nearest neighbors:

- Small k → sensitive to noise  
- Large k → smoother decisions  

The script tests different values of k to select the one with the highest accuracy on the test set. With the chosen k, KNN can achieve **over 97% accuracy** on the digits dataset.

---

## 3. Logistic Regression

Two approaches are explored:

### 3.1 Regularization parameter C

Different values of the regularization parameter C are tested:

- **C = 1** (default)
- **C = 100** (less regularization, more flexible)
- **C = 0.01** (more regularization, simpler model)

Higher C values produce more complex models, while lower C values produce simpler models. The best balance is typically around C=1, achieving 96–97% accuracy.

### 3.2 Multiclass classification (One-vs-Rest)

Logistic regression is applied in a **One-vs-Rest** strategy to handle multiple classes (digits 0–9).  
Predictions are compared with the true labels, and the accuracy, classification report, and confusion matrix are evaluated.

---

## 4. Cross-Validation

Cross-validation is used to evaluate model performance more reliably than a single train/test split.  
**Stratified K-Folds** ensure that each fold maintains the class distribution.  
This method improves mean accuracy by reducing variability caused by a single split.

---

## 5. Grid Search for Hyperparameter Tuning

Grid search is used to find the best combination of hyperparameters (e.g., C and penalty type) using cross-validation.  
- **L1 (Lasso)**: some coefficients are set to 0 → automatic feature selection  
- **L2 (Ridge)**: all coefficients are shrunk → more stable solutions  

The best combination is selected based on mean cross-validation accuracy, and the final model is evaluated on the independent test set. This ensures proper generalization.

---

## 6. Grid Search with Separate Validation Set

To avoid information leakage from the test set, the dataset is split into three parts:

1. **Training set** – to train the model  
2. **Validation set** – to tune hyperparameters  
3. **Test set** – to evaluate the final performance  

Hyperparameter tuning is done on the training and validation sets, and the test set provides an unbiased estimate of performance.

---

## 7. Confusion Matrix and Predictions

The final model is evaluated using a **confusion matrix** to visualize prediction errors. Most predictions are correct, with misclassifications mainly occurring in similar digits.  

Visual examples of predictions allow comparing the model's predicted label with the true label, helping to interpret model performance.

---

## 8. Conclusion

This script demonstrates a full workflow for **classification using Scikit-learn**:

- Data preparation, normalization, and visualization  
- KNN and logistic regression for multiclass classification  
- Cross-validation and GridSearchCV for reliable hyperparameter tuning  
- Model evaluation using accuracy, confusion matrix, and classification report  

It serves as a comprehensive introduction to machine learning with Scikit-learn using image-based datasets.
