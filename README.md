**Project Introduction :** An End-to-End Machine Learning Pipeline for Predicting Customer Attrition

**Project Overview :**
This project aims to predict customer churn for a telecommunications company using the Telco Customer Churn dataset. The goal is to identify customers at high risk of leaving the service, allowing the company to implement proactive retention strategies.
This is not just a modeling project; it is a complete ML Pipeline that handles everything from raw data ingestion to model interpretability.

**Technical Workflow**

1 .**Exploratory Data Analysis (EDA):**
* **Analysis of numerical features:** Distribution of numerical features using `KDE` and `Boxplots` to identify skewness and outliers.
* **Analysis of categorical features:** Examining the relationship between categorical features and the target variable (customer churn) using `stacked bar charts` and `count plots`.
* **Multivariate Analysis:** Analyzing correlations between numerical features using `heatmaps` and `pairplots` to understand the interactions among the features.
* **Outlier Detection: Implemented `Robust Z-score` (based on Median Absolute Deviation) to handle outliers more effectively than standard `Z-score`.**

2 .**Data Preprocessing & Feature Engineering**
* **Cross Validation:** To ensure the model's generalizability and prevent `overfitting` and uniforming the distribution of classes in fold using `StratifiedKFold`
* **PipeLine:** Using a `pipeline` for modeling to prevent data leakage.
* **Data Cleaning:** Handling missing values using `SimpleImputer` With the `median` strategy for numerical features and the `most frequent` strategy for categorical features.
* **Advanced Pipeline:** `ColumnTransformer` is used to simultaneously apply different transformations `StandardScaler` for numerical features and `one-hot encoding` for categorical features.
* **Handling Imbalance:** Using the `SMOTE` technique within an `ImbPipeline` to prevent data leakage and manage imbalanced data.
* **Feature Selection:** In each pipeline, the `SelectFromModel` method was used during the feature selection stage to ensure that only the most influential features were included.

3 .**Modeling and Evaluation:**
* **Comparison of the performance of several powerful algorithms:**
* Decision Tree (DTC)
* Random Forest (RFC)
* SVM
* XGBoost
* CatBoost

4 .**Results & Performance**
* **Comparison of the performance of five powerful models on test data based on accuracy, precision, recall, F1, and AUC metrics.**
* models  |  accuracy_test  | precision_test   | recall_test  | f1_test   | auc_test
* DTC	  | 0.708590	    | 0.471921	       | 0.859964	  | 0.609415  | 0.832743
* RFC	  | 0.745610	    | 0.512281	       | 0.786355	  | 0.620397  | 0.838330
* SVM	  | 0.743237        | 0.509368	       | 0.780969	  | 0.616584  | 0.811648
* XGC     | 0.740864	    | 0.506403	       | 0.780969	  | 0.614407  | 0.834672
* CatBoost| 0.743237        | 0.509346	       | 0.782765	  | 0.617127  | 0.836852

Techniques such as StandardScaler for scaling numerical features, SMOTE for managing class balance, and SelectFromModel for identifying influential features have stabilized model performance and played a decisive role in preventing overfitting.

**Selecting the final model:** Based on the results of various evaluations conducted on five `RFC` algorithm models, the model selected for the customer churn prediction scenario was chosen by considering the balance between `precision` and `recall`.

5 .**Model Interpretability**
To better understand the model's decisions, the `Permutation Importance` method was used to determine which features have the greatest impact on predicting customer churn.

6 .**Model Interpretability Analysis (Partial Dependence)**
I used `PDP` to visualize the correlation between the impact of key features and the model's target variable.

**Technologies Used:**
* **Language:** `Python`
* **Libraries:** `Pandas` , `NumPy` , `Matplotlib` , `Seaborn`
* **Machine Learning:** `Scikit-learn` , `XGBoost` , `CatBoost` , `Imbalanced-learn`
