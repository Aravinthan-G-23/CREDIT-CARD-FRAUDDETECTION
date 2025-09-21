import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import VarianceThreshold

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import GridSearchCV

import joblib


fraudtrain = pd.read_csv("fraudTrain.csv")
fraudtest = pd.read_csv("fraudTest.csv")

print(fraudtrain.head())
print(fraudtest.head())
print(fraudtrain.isnull().sum())
print(fraudtest.isnull().sum())
print(fraudtrain.columns)
print(fraudtest.columns)
print(fraudtrain['is_fraud'].value_counts())
print(fraudtest['is_fraud'].value_counts())
fraudtrain.describe()
fraudtest.describe()
df_combined = pd.concat([fraudtrain, fraudtest], axis=0, ignore_index=True)
df_combined.head()
df_combined.shape
df_combined.size
df_combined.info()

columns_to_drop = ["first", "last", "job", "dob", "trans_num", "street", 
                   "trans_date_trans_time", "city", "state"]


df_combined = df_combined.drop(columns=[col for col in columns_to_drop if col in df_combined.columns])


print(df_combined.head())

sns.countplot(x='gender', data=df_combined, color='orange')
plt.title('GENDER DISTRIBUTION')
plt.xticks(rotation=90)
plt.show()   
print(df_combined.columns)

df_combined_numeric = df_combined.select_dtypes(include=['number'])
correlation_matrix = df_combined_numeric.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", cbar=True)
plt.title("Correlation Matrix")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Prepare data
X = df_combined.drop("is_fraud", axis=1)
y = df_combined["is_fraud"]

# Encode categorical variables
X = X.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtypes == 'object' else col)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = map(StandardScaler().fit_transform, [X_train, X_test])

# Train and predict
r_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_pred = r_model.predict(X_test)

# Evaluate
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
