import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Load the train and test data
try:
    train_data = pd.read_csv("test_data.txt", sep=':::', header=None, engine='python')
    test_data = pd.read_csv("train_data.txt", sep=':::', header=None, engine='python')
    
    # Renaming the columns to make the dataset more readable
    train_data.columns = ['SI.NO', 'MOVIE', 'MOVIETYPE','SUMMARY']
    test_data.columns = ['SI.NO', 'MOVIE', 'SUMMARY']
    with open("train_data.txt", "r") as file:
        print(file.readlines()[:5])  # Print the first 5 lines

    
    # Displaying the first few rows of train_data
    print(train_data.head())

except Exception as e:

    print(f"Error: {e}")
train_data.info()
train_data.describe()
train_data.isnull().sum()
test_data.isnull().sum()
train_data.count()
train_data.iloc[0:3]
train_data.loc[0]
train_data.shape
test_data.shape
print(train_data.columns)
print(train_data.head()) 
model = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=5000),  # TF-IDF Vectorization
    LogisticRegression(max_iter=1000)  # Logistic Regression Classifier
)

# Alternatively, you could use other classifiers like SVC (Support Vector Classifier)
# model = make_pipeline(
#     TfidfVectorizer(stop_words='english', max_features=5000),
#     SVC(kernel='linear')
# )

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)

# Evaluate the performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

sns.countplot(x="MOVIETYPE", data=train_data)
plt.xlabel('Movie Category')
plt.ylabel('Count')
plt.title('Movie Genre Plot')
plt.xticks(rotation=90)
plt.show()
sns.displot(train_data.MOVIETYPE, kde=True, color="black")
plt.xticks(rotation=98)
plt.figure(figsize = (14,10))
count1=train_data.MOVIETYPE.value_counts()
sns.barplot(x=count1,y=count1.index,orient='h',color='Blue')
plt.xlabel("count")
plt.xlabel("Movie type")
plt.title("Movie Genre plot")
plt.xticks(rotation=90);
plt.show()
