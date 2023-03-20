import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn .model_selection import train_test_split
from sklearn import metrics

col_names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
df = pd.read_csv("diabetes.csv",
names = col_names
).iloc[1:]
print(df.head())

features = ['pregnant','insulin','bmi','age','glucose','pedigree','bp']
X = df[features]
y = df.label

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 1)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))