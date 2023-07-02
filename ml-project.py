#%%
#1.libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#2.data preprocessing
#2.1. data loading
data = pd.read_csv('test.csv')
data.head(3)
data = data.drop(columns=['engine', 'engine_size', 'automatic_transmission',
                          'drivetrain', 'min_mpg', 'max_mpg', 'damaged', 'first_owner',
                          'personal_using', 'turbo', 'alloy_wheels',
                          'adaptive_cruise_control', 'navigation_system',
                          'power_liftgate', 'backup_camera', 'keyless_start',
                          'remote_start', 'sunroof/moonroof', 
                          'automatic_emergency_braking', 'stability_control',
                          'leather_seats', 'memory_seat', 'third_row_seating',
                          'apple_car_play/android_auto', 'bluetooth',
                          'usb_port', 'heated_seats', 'interior_color', 'exterior_color'])

#checking data types
print(data.dtypes)
#checking for empty rows
data.isna().sum()
#filling empty rows with median values
for column in ['year', 'mileage', 'price']:
    data[column] = data[column].fillna(data[column].median())
data.isna().sum()
for column in ['brand', 'model', 'transmission', 'fuel_type']:
    data[column] = data[column].fillna(data[column].mode()[0])
data.isna().sum()

#Visualization
plt.figure(figsize=[45,5])
plt.subplot(1,3,1)
sns.barplot(data=data, x = 'fuel_type', y = 'price')
plt.title('Fuel Type vs Price')
plt.xlabel('Fuel Type')
plt.ylabel('Price')
plt.show()


plt.figure(figsize=[75,5])
plt.subplot(1,3,3)
sns.barplot(data=data, x = 'brand', y = 'price')
plt.title('Brand vs Price')
plt.xlabel('Car Brand')
plt.ylabel('Price')
plt.show()

#Correlation between different variables
corr = data[['price', 'mileage', 'year']].corr()
plt.title('Heatmap')
sns.heatmap(corr, annot=True)
plt.show()

sns.pairplot(data[['price', 'mileage', 'year']])
plt.show()

#%%
#Label encoding
from sklearn import preprocessing
data = data.drop(columns=['transmission'])
le = preprocessing.LabelEncoder()
data.iloc[:, 1] = le.fit_transform(data.iloc[:,1])
data.iloc[:, 0] = le.fit_transform(data.iloc[:,0])
data.iloc[:, 4] = le.fit_transform(data.iloc[:,4])

#odel = model.reshape(-1,1)
#OneHotEncoder
#ohe = preprocessing.OneHotEncoder()
#model = ohe.fit_transform(model).toarray()


#%%
#data splitting
y = data.iloc[:, -1]
data = data.drop(columns=['price'])
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(data,y,test_size=0.33, random_state=0)
#%%
#data scaling
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


#%%
#K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

plt.title('KNeighbors Classifier')
plt.scatter(y_pred, y_test)
#%%
#Support Vector
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

plt.title('Support Vector Classifier')
plt.scatter(y_pred, y_test)
#%%
#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
plt.title('Gaussian Naive Bayes')
plt.scatter(y_pred, y_test)

#%%
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
plt.title('Decision Tree Classifier')
plt.scatter(y_pred, y_test)
#%%
#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=15, criterion = 'gini')

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
plt.title('Random Forest Classifier')
plt.scatter(y_pred, y_test)
# %%
