# فرا خوانی پکیج های مورد نیاز
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# بارگزاری دیتاست از فایل CSV
df_customers = pd.read_csv("Telco-Customer-Churn.csv")

# نمایش پنج سطر اول دیتاست جهت آشنایی اولیه
print(df_customers.head())

# بررسی ابعاد داده
print(df_customers.shape)

# نمایش نام ستون ها
print(df_customers.columns)

# نمایش نوع داده ستون ها
print(df_customers.dtypes)

# نمایش مقادیر یکتا ستون ها
print(df_customers.nunique())

# نمایش نوع داده ستون ها و تعداد مقادیر غیر خالی
print(df_customers.info())

# جایگزاری فضا های خالی با مقدار nan با هدف مدیریت بهتر missing vlues
df_customers = df_customers.replace(" ", np.nan)

# تبدیل نوع داده ستون TotalCharges به مقادیر float
df_customers["TotalCharges"] = df_customers["TotalCharges"].astype(float)

# تبدیل نوع داده ستون SeniorCitizen به مقادیر str
df_customers["SeniorCitizen"] = df_customers["SeniorCitizen"].astype(str)

# حذف ستون customerID چونکه برای آموزش مدل کاربردی ندارد
df_customers.drop(columns="customerID", inplace=True)

# بررسی تعداد سطر های تکراری
df_customers.duplicated().sum()

# بررسی تعداد ستون های تکراری
duplicated_columns = df_customers.T.duplicated()
print(df_customers.columns[duplicated_columns])

# حذف سطر های تکراری
df_customers.drop_duplicates(inplace=True)

# بررسی تعداد مقادیر گمشده هر ستون
df_customers.isnull().sum()

# تفکیک ویژگی های عددی
numerical_features = df_customers.select_dtypes(["int64", "float64"])

# تفکیک ویژگی های دسته ای
category_features = df_customers.select_dtypes(object)

# رسم نمودار Boxplot برای شناسایی مقادیر پرت در ویژگی های عددی
for column in numerical_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=numerical_features, y=column)
    plt.grid()
    plt.show()

# تابع robust z-score برای شناسایی ردیف هایی که دارای مقادیر پرت هستند
def robust_zscore(data=None, columns=None):
    medians = np.median(data[columns], axis=0)
    mad = np.median(np.abs(data[columns] - medians), axis=0)
    mi = (0.6745 * (data[columns] - medians)) / mad

    z_threshold = 3

    outlier_detect = np.array(np.abs(mi) > z_threshold)

    outlier_indices = data[columns].index[outlier_detect.any(axis=1)]
    return outlier_indices

print(robust_zscore(data=df_customers, columns=["tenure", "MonthlyCharges", "TotalCharges"]))

# نمایش آماری ویژگی های عددی
numerical_features.describe()

# تحلیل توضیع داده ها با استفاده آمار توصیفی
for column in numerical_features:
    print(column)
    print(f"mean => {numerical_features[column].mean()}, median => {numerical_features[column].median()}")
    print(f"variance => {numerical_features[column].var()}, standard deviation => {numerical_features[column].std()}")
    print(f"skewness => {numerical_features[column].skew()}, kurtosis => {numerical_features[column].kurt()}")
    print("." * 100)

numerical_features["Churn"] = df_customers["Churn"].map({"No": 0, "Yes": 1})

for column in numerical_features:
    if column == "Churn":
        continue

    # ایجاد اشکال
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # نمایش توزیع ویژگی های عددی با استفاده از نمودار KDE
    sns.kdeplot(data=numerical_features, x=column, color='skyblue', fill=True, ax=axes[0])

    # ۳. محاسبه میانگین و میانه
    mean_val = numerical_features[column].mean()
    median_val = numerical_features[column].median()

    # ۴. رسم خطوط عمودی برای میانگین و میانه
    axes[0].axvline(mean_val, color='red', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[0].axvline(median_val, color='green', linewidth=2, label=f'Median: {median_val:.2f}')

    axes[0].set_title(f'Distribution of {column} with Mean & Median', fontsize=15)
    axes[0].grid()

    # نمایش توزیع ویژگی های عددی بر اساس طبقه بندی دو کلاسه با استفاده از نمودار KDE
    sns.kdeplot(data=numerical_features, x=column, color='skyblue', fill=True, hue="Churn", ax=axes[1])
    axes[1].set_title(f'Distribution of the {column} feature based on binary classification', fontsize=15)
    axes[1].grid()

    plt.tight_layout()
    plt.show()

# تحلیل همبستگی ویژگی های عددی با استفاده از نمودار heatmap
sns.heatmap(data=numerical_features.corr(method="pearson"), annot=True, cbar=False)
plt.show()

# رسم جفت نمودارها pairplot
sns.pairplot(numerical_features)
plt.show()

# تحلیل توزیع کلاس ها در ویژگی های دسته ای با استفاده از نمودار های CountPlot و PiePlot
for column in category_features.columns:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # رسم نمودار CountPlot روی محور اول
    sns.countplot(data=category_features, x=column, hue="Churn", ax=axes[0])
    axes[0].set_title(f'Distribution of {column}', fontsize=14)
    axes[0].grid()

    counts = category_features[column].value_counts()

    # رسم نمودار PiePlot روی محور دوم
    axes[1].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    axes[1].set_title(f'Proportion of {column}', fontsize=14)

    # تنظیم فاصله و نمایش
    plt.tight_layout()
    plt.show()

# نمایش توزیغ کلاس ها در ویژگی های دسته ای با استفاده از Stacked Bar Plot
for column in category_features.columns:
    if column == 'Churn':
        continue

    plt.figure(figsize=(10, 5))

    df_plot = pd.crosstab(category_features[column], category_features['Churn'])

    df_plot.plot(kind='bar', stacked=True, ax=plt.gca())

    plt.title(f'Churn Rate by {column}', fontsize=14)
    plt.ylabel('Proportion')
    plt.xlabel(column)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

# بررسی توازن نمونه های متعلق به کلاس های 0 & 1
print((df_customers["Churn"].value_counts() / len(df_customers)) * 100)

plt.pie(df_customers["Churn"].value_counts(), autopct='%1.1f%%')
plt.show()

# آماده سازی متغیر هدف Target
y = df_customers["Churn"].map({"No": 0, "Yes": 1}).values.reshape(-1)

# حذف متغیر هدف Target
X = df_customers.drop(columns="Churn")

# تعریف لیستی از ویژگی های عددی و دسته ای برای انجام مراحل پیش پردازش
num_features = ["tenure", "MonthlyCharges", "TotalCharges"]

cat_features = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]

warnings.filterwarnings("ignore")

# فراخوانی پیکیج های مورد نیاز برای مدل سازی
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

# خط لوله پیش پردازش برای ویژگی های عددی (پر کردن مقادیر گمشده با استفاده از استراتژی میانه و استاندارد سازی ویژگی های عددی)
numeric_transform = Pipeline(steps=[
    ("missing_values", SimpleImputer(strategy='median')),
    ("standard_scaler", StandardScaler())
])

# خط لوله پیش پردازش برای ویژگی های دسته ای (پر کردن مقادیر گمشده با استفاده از استراتژی most_frequent و encode کردن ویژگی های دسته ای )
category_transform = Pipeline(steps=[
    ("missing_values", SimpleImputer(strategy='most_frequent')),
    ("one_hot_encoder", OneHotEncoder())
])

# انجام همزمان فرایند پیش پردازش روی ویژگی های عددی و دسته ای
preprocessor = ColumnTransformer(transformers=[
    ("num_transform", numeric_transform, num_features),
    ("cat_transform", category_transform, cat_features)
])


pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=1234)),
    ("classifier", LogisticRegression()),
])

# استفاده از StratifiedKFold با هدف تعادل نسبی کلاس ها در هر fold
Stratify_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

# انجام اعتبار سنجی با استفاده از cross validation
cross_validation = cross_validate(estimator=pipeline, X=X, y=y, verbose=10, cv=Stratify_fold, scoring=["accuracy", "precision", "recall", "f1"])

# تعریف خط لوله ها برای چهار مدل طبقه بند (DTC, RFC, SVM, XGBOOST) برای مقایسه عملکردی
# هر مدل دارای انتخاب ویژگی با select from model و متعادل سازی داده ها با smote
pipelines = {
        "DTC": ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectFromModel(LogisticRegression(random_state=1234))),
            ("smote", SMOTE(random_state=1234)),
            ("DTClassifier", DecisionTreeClassifier(max_depth=5 ,random_state=1234)),
        ]),

        "RFC": ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectFromModel(LogisticRegression(random_state=1234))),
            ("smote", SMOTE(random_state=1234)),
            ("RFClassifier", RandomForestClassifier(max_depth=5 ,random_state=1234)),
        ]),

        "SVM": ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectFromModel(LogisticRegression(random_state=1234))),
            ("smote", SMOTE(random_state=1234)),
            ("SVClassifier", SVC(probability=True,random_state=1234)),
        ]),

        "XGBoost": ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectFromModel(LogisticRegression(random_state=1234))),
            ("smote", SMOTE(random_state=1234)),
            ("XGBoost classifier", XGBClassifier(max_depth=5, learning_rate=0.01, random_state=1234)),
        ]),

        "CatBoost": ImbPipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("feature_selection", SelectFromModel(LogisticRegression(random_state=1234))),
                    ("smote", SMOTE(random_state=1234)),
                    ("CatBoost classifier", CatBoostClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=1234)),
                ]),
}

# تفکیک داده های آموزشی train و آزمون test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1234)

result_train = {}
result_test = {}

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_predict_train = pipe.predict(X_train)
    y_predict_test = pipe.predict(X_test)

    y_proba_train = pipe.predict_proba(X_train)[:, 1]
    y_proba_test = pipe.predict_proba(X_test)[:, 1]

    result_train[name] = {
        "accuracy_train": accuracy_score(y_train, y_predict_train),
        "precision_train": precision_score(y_train, y_predict_train),
        "recall_train": recall_score(y_train, y_predict_train),
        "f1_train": f1_score(y_train, y_predict_train),
        "auc_train": roc_auc_score(y_train, y_proba_train),
    }

    result_test[name] = {
        "accuracy_test": accuracy_score(y_test, y_predict_test),
        "precision_test": precision_score(y_test, y_predict_test),
        "recall_test": recall_score(y_test, y_predict_test),
        "f1_test": f1_score(y_test, y_predict_test),
        "auc_test": roc_auc_score(y_test, y_proba_test),
    }


# تبدیل نتایج train و test به DataFrame
results_train = pd.DataFrame(result_train).T
results_test = pd.DataFrame(result_test).T

print(results_train)
print(results_test)

# رسم نمودارهای ستونی برای مقایسه عملکرد مدل‌ها در داده‌های آموزش و تست
n_cols = 3
n_rows = (len(results_train) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

for i, column in enumerate(results_train):
    sns.barplot(results_train[column], ax=axes[i])
    axes[i].set_title(column)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

n_cols = 3
n_rows = (len(results_test) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

for i, column in enumerate(results_test):
    sns.barplot(results_test[column], ax=axes[i])
    axes[i].set_title(column)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

model_names = list(pipelines.keys())

all_importance_results = []

for name in model_names:
    current_pipeline = pipelines[name]

    # محاسبه میزان اهمیت ویژگی ها با استفاده از permutation_importance
    result = permutation_importance(
        current_pipeline, X_test, y_test,
        n_repeats=5,
        random_state=1234,
        n_jobs=-1
    )

    # دریافت نام ویژگی ها از دیتاست test
    feature_names = X_test.columns

    # ذخیره نتایج اهمیت ویژگی ها از نظر مدل ها
    model_df = pd.DataFrame({
        'Model': name,
        'Feature': feature_names,
        'Importance_Mean': result.importances_mean,
    })

    all_importance_results.append(model_df)

for model in all_importance_results:
    # مرتب‌سازی ویژگی‌ها بر اساس بیشترین اهمیت برای خوانایی بهتر نمودار
    df_sorted = model.sort_values(by='Importance_Mean', ascending=False)

    plt.figure(figsize=(12, 6))

    # رسم نمودار barplot برای نمایش میزان اهمیت و تاثیر گزاری ویژگی ها روی تصمیم گیری مدل
    sns.barplot(x="Importance_Mean", y="Feature", data=df_sorted, palette="viridis")

    plt.title(f"Feature Importance: {df_sorted['Model'].iloc[0]}")
    plt.xlabel("Mean Importance Score")
    plt.ylabel("Features")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# استخراج مراحل پیش پردازش انتخاب ویژگی و مدل سازی از داخل پایت لاین
preprocessor = pipelines["RFC"].named_steps["preprocessor"]
selector = pipelines["RFC"].named_steps["feature_selection"]
estimator = pipelines["RFC"] .named_steps["RFClassifier"]

# دریافت نام تمام ستون‌های تولید شده توسط پیش‌پردازشگر خط لوله
new_feature_names = preprocessor.get_feature_names_out()

# پیدا کردن ایندکس ستون‌هایی که توسط مرحله انتخاب ویژگی تایید شده‌اند
selected_indices = selector.get_support(indices=True)

# فیلتر کردن نام ستون‌های اصلی بر اساس ستون‌های انتخاب شده
# در نهایت فقط نام ستون‌هایی را داریم که واقعاً وارد مدل شده‌اند
feature_names = [new_feature_names[i] for i in selected_indices]

# اعمال فرایند پیش پردازش روی داده های test
X_preprocess = preprocessor.transform(X_test)

# اعمال فرایند انتخاب ویژگی روی داده های test
X_selector = selector.transform(X_preprocess)

# ترسیم نمودار های Partial Dependence برای ویژگی های منتخب
fig, ax = plt.subplots(figsize=(15, 15))
display = PartialDependenceDisplay.from_estimator(estimator=estimator, X=X_selector, features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], feature_names=feature_names, ax=ax)
plt.show()