# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:31:44 2018

@author: jashs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('train.csv')

print(dataset.shape)
print(dataset.count())

print(dataset.head())


y_train=dataset.iloc[:,1].values
#X_train=dataset.iloc[:,[0,2,3,4,5,6,7,8,9,10,11]].values
X_train=dataset.iloc[:,[2,4,5,6,7,9,11]].values

#X_train consists of Pclass,Sex,Age,SibSp,Parch,Fare,Embarked

print (X_train.shape)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:,2:3])
X_train[:,2:3] = imputer.transform(X_train[:, 2:3])


#imputer2 = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
#imputer2 = imputer2.fit(X_train[:,6:7])
#X_train[:,6:7] = imputer2.transform(X_train[:, 6,7])
#Above code didn't work. Cannot convert string to float, so we'll do it manually for now
X_train[61,6]='S'
X_train[829,6]='S'
    
#Dealing with Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
X_train[:, 6] = labelencoder_X.fit_transform(X_train[:, 6])

onehotencoder = OneHotEncoder(categorical_features = [6])
X_train = onehotencoder.fit_transform(X_train).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)



#Importing the test data
test_data=pd.read_csv('test.csv')

X_test=test_data.iloc[:,[1,3,4,5,6,8,10]].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer_test = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_test = imputer_test.fit(X_test[:,2:3])
X_test[:,2:3] = imputer_test.transform(X_test[:, 2:3])

imputer_test2 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_test2 = imputer_test2.fit(X_test[:,5:6])
X_test[:,5:6] = imputer_test2.transform(X_test[:, 5:6])


#Dealing with Categorical Data
labelencoder_X_test = LabelEncoder()
X_test[:, 1] = labelencoder_X_test.fit_transform(X_test[:, 1])
X_test[:, 6] = labelencoder_X_test.fit_transform(X_test[:, 6])

onehotencoder_test = OneHotEncoder(categorical_features = [6])
X_test = onehotencoder_test.fit_transform(X_test).toarray()

# Predicting the Test set results
y_pred = classifier.predict(X_test)

p_id=test_data.iloc[:,0].values
a=np.array(p_id)
b=np.array(y_pred)
final_pred=np.column_stack((a,b))

new_df=pd.DataFrame(final_pred)

new_df.to_csv('predictions.csv')





