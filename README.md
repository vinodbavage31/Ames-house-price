# House Price Prediction using Machine Learning

## Project Overview
This project implements an end-to-end machine learning pipeline to predict house prices using structured tabular data.  
The focus of this work is on **clean data preparation, reproducible preprocessing, robust model validation, and interpretability**, rather than model complexity.

The final model achieves strong predictive performance through careful data handling and a clean modeling pipeline.

---

## Dataset
- Source: Kaggle
- Type: Tabular housing data with numerical and categorical features
- Target Variable: `SalePrice`

The dataset contains:
- Missing values
- Numeric-coded categorical features
- Meaningful absence indicators (e.g., `None` vs `0`)
- Skewed target distribution

---

## Workflow Summary

### 1. Data Inspection
- Examined dataset shape, feature types, and summary statistics
- Identified missing values and invalid placeholders
- Verified semantic correctness of feature data types

### 2. Data Cleaning
- Handled missing values based on feature meaning
- Distinguished between:
  - True missing values (`NaN`)
  - Meaningful absence (e.g., no basement, no garage)
  - Valid zero values
- Converted numeric-coded categorical features to categorical type

### 3. Feature Engineering
- Separated numerical and categorical features
- Applied:
  - Standard scaling to numerical features
  - One-hot encoding to categorical features
- Used `ColumnTransformer` to ensure consistent preprocessing

### 4. Modeling Approach
- Train–test split performed **before preprocessing** to avoid data leakage
- Built a unified pipeline combining preprocessing and modeling
- Models evaluated:
  - Linear Regression
  - Ridge Regression
- Applied log transformation to the target variable to handle skewness

### 5. Model Evaluation
Models were evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Train vs test performance comparison to assess generalization

---

## Model Performance

### Linear Regression (Final Model)
- **MAE:** 15,074.80  
- **RMSE:** 22,871.29  
- **R² Score:** 0.9318  

### Ridge Regression
- **MAE:** 15,747.31  
- **RMSE:** 23,852.03  
- **R² Score:** 0.9258  

### Generalization Check
- **Train R²:** 0.9494  
- **Test R²:** 0.9318  

The small performance gap indicates strong generalization with no significant overfitting.

---
## Model Improvement
- Initial baseline model achieved approximately **R² ≈ 0.88**
- Performance improved to **R² ≈ 0.93** through:
  - Proper data cleaning
  - Correct feature typing
  - Leakage-free preprocessing
  - Log transformation of the target variable

---

## Feature Importance
Feature importance was derived from model coefficients after preprocessing.  
This analysis provides insight into which features most strongly influence house prices and in what direction.

---

## Key Takeaways
- Strong performance can be achieved with simple models when data is handled correctly
- Clean pipelines and proper validation are more impactful than model complexity
- Interpretability and reproducibility are essential for production-ready models

---

## Tools and Libraries
- Python
- pandas
- NumPy
- scikit-learn
- matplotlib

---

## Future Work
- Hyperparameter tuning with cross-validation
- Feature interaction engineering
- Non-linear models (e.g., Gradient Boosting)
- Model deployment and monitoring

---

## Author
This project was developed as a learning-focused, professional machine learning workflow emphasizing clarity, correctness, and best practices.
