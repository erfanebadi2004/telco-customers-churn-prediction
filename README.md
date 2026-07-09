# Telco Customer Churn Prediction

## Project Overview
An end-to-end Machine Learning pipeline designed to predict customer churn in the telecommunications sector. This project goes beyond simple modeling; it encompasses a complete data-to-insight workflow, focusing on identifying high-risk customers to enable proactive retention strategies.

## Source
The source of the received data is `Kaggle`.

## Project Workflow
### 1. Exploratory Data Analysis (EDA)
- Analysis distribution numerical features using `KDE` and `Box plots`.
- Multivariate analysis using `Heatmaps` and `Pairplots`.
- Analyzing categorical variables using `count` and `pie charts`.
- **Advanced Outlier Detection:** Utilizing `Robust Z-score` for better accuracy

### 2. Data Preprocessing & Pipeline Architecture
- **Handling Missing Values :** Using `SimpleImputer` with strategy `Median` for numerical features and strategy `most frequent` for categorical features
- **Feature Engineering :** Using `StandarsScaler` for numerical features and Using `OneHotEncoder` for Categorical features
- **Leakage Prevention :** Using `Pipeline` and `ColumnTransformer`.
- **Handling Imbalance :** Using `SMOTE` within `ImbPipeline`.
- **Feature Selection :** Leveraging `SelectFromModel`.

### 3. Modeling & Evaluation
We compared five powerful algorithms to determine the best fit for this imbalanced classification problem:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Decision Tree | 0.7086 | 0.4719 | 0.8599 | 0.6094 | 0.8327 |
| **Random Forest** | 0.7456 | 0.5123 | 0.7864 | 0.6204 | **0.8383** |
| SVM | 0.7432 | 0.5094 | 0.7810 | 0.6166 | 0.8116 |
| XGBoost | 0.7409 | 0.5064 | 0.7810 | 0.6144 | 0.8347 |
| CatBoost | 0.7432 | 0.5093 | 0.7828 | 0.6171 | 0.8369 |

## 4.Final Model Selection
`Random Forest` was selected as the final model because it achieved the highest `ROC-AUC` while maintaining a balanced tradeoff between `precision` and `recall`.

## 5.Model Interpretability
To ensure the model is not a "black box," we analyzed feature influence:
- **Permutation Importance:** Identifying the primary drivers of churn.
- **Partial Dependence Plots (PDP):** Visualizing the relationship between features and the churn target variable.

## Technologies
- **Language:** `Python`
- **Analysis:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `XGBoost`, `CatBoost`, `Imbalanced-learn`

*Developed by [Bktash Ebadi]
