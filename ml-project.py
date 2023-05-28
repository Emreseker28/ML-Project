#%%
#1.libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.data preprocessing
#2.1. data loading
data = pd.read_csv('honda_sell_data.csv')
year = data.iloc[:,0]
model = data.iloc[:, 2]
price = data.iloc[:, 4].values
x = pd.concat([model, year], axis=1)
#test
print(data)

#%%
#Label encoding
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
x.iloc[:, 0] = le.fit_transform(data.iloc[:,2])

#odel = model.reshape(-1,1)
#OneHotEncoder
#ohe = preprocessing.OneHotEncoder()
#model = ohe.fit_transform(model).toarray()
y = price

#%%
#data splitting
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)
#%%
#data scaling
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#%%
#Logistic regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
#print(y_pred)
#print(y_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


#%%
#K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)


#%%
#Support Vector
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)

#%%
#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')

#%%
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('DTC')
print(cm)

#%%
#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=15, criterion = 'gini')

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('RFC')
print(cm)

# %%
