import pandas as pd

# Specify encoding
df = pd.read_csv("spam.csv", encoding='latin1')


print("Columns in the dataframe:")
print(df.columns)

print("\nFirst few rows of the dataset:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types of each column:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())
df.info()
print("Number of Rows:",df.shape[0])
print("Number of columns:",df.shape[1])
df.isnull().sum()
df.drop(columns=df[['Unnamed: 2','Unnamed: 3','Unnamed: 4']], axis=1, inplace=True)
df.head()
df.columns = ['spam/ham', 'sms']
df.loc[df['spam/ham'] == 'spam', 'spam/ham'] = 0
x=df.sms
x.head()
y=df['spam/ham']
y.head()
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

print(x.shape)
print(xtrain.shape)
print(xtest.shape)
xtrain,xtest
ytrain,ytest
from sklearn.feature_extraction.text import TfidfVectorizer

feat_vect = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
print(feat_vect)
df.loc[df['spam/ham'] == 'ham', 'spam/ham'] = 1
df.head()
ytrain = ytrain.astype('int')
ytest = ytest.astype('int')
xtrain_vec = feat_vect.fit_transform(xtrain)
xtest_vec = feat_vect.transform(xtest)
print(xtrain)
xtrain_vec
print(xtrain_vec)
print(xtest_vec)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(xtrain_vec, ytrain)
print(lr)
lr.score(xtrain_vec, ytrain)
0.9694862014808167

lr.score(xtest_vec, ytest)
0.9524663677130045

pred_lr = lr.predict(xtest_vec)
pred_lr
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
accuracy_score(ytest,pred_lr)
confusion_matrix(ytest,pred_lr)
print(classification_report(ytest,pred_lr))
