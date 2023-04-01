# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:20:02 2022

@author: Dr. Sony
"""

## PROSTATE CANCER 

#Importing necessary packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

#========================================================================================
# Reading the data

data = pd.read_csv('prostate.txt',sep='\t',index_col=0)

# Structure of the data
data.info()

# Summary of the data
describe=data.describe()

#=========================================================================================

# DATA CLEANING

# Dropping column 'train' with no values
data=data.drop(['train'],axis=1)

# No. of missing values in each column
data.isnull().sum()
#no missing values present

# Checking for  duplicate records
len(data[data.duplicated(subset = None, keep = "first")]) 
#no duplicate records found

#Checking frequency of categories
svi= data['svi'].value_counts().sort_index()
gleason=data['gleason'].value_counts().sort_index()

#Changing data type
data['svi'] = data.svi.astype(object)
data['gleason'] = data.gleason.astype(object)
#svi and gleason showed categorical values and were changed to object data type

#==========================================================================================

# DATA VISUALIZATION

plt.subplots(figsize=(8,6))

# lcp vs lpsa
sns.scatterplot(x='lcp',y='lpsa',data=data)
plt.show()
#Increase in lcp saw an increase in lpsa value


# lcavol vs lpsa
sns.scatterplot(x='lcavol',y='lpsa',data=data)
plt.show()
# lcavol showed v high dependency on lpsa => +ve


# lweight vs lpsa
sns.scatterplot(x='lweight',y='lpsa',data=data)
plt.show()
# no clear trend => not much impact on lpsa

# age vs lpsa
sns.scatterplot(x='age',y='lpsa',data=data)
plt.show()
# 60-70 yrs of men are seen more with lpsa score

# lbph vs lpsa
sns.scatterplot(x='lbph',y='lpsa',data=data)
plt.show()
# no trend hence lbph does not impact lpsa values in any sense

#pgg45 vs lpsa
sns.scatterplot(x='pgg45',y='lpsa',data=data)
plt.show()

#no clear trend was apparent

##Boxplot between lpsa and svi
sns.boxplot(x='svi',y='lpsa',data=data)
plt.show()
pd.crosstab(data['svi'], columns = 'count', normalize = True)
#78% have svi the rest 21.6% do not
#graph shows that men with svi have higher lpsa

#Boxplot between gleason and lpsa
sns.boxplot(x='gleason',y='lpsa',data=data)
plt.show()
pd.crosstab(data['gleason'], columns = 'count', normalize = True)
#lpsa doesnt show a clear trend with gleason but showed an increase 
# can conclude that it has some impact but has to be checked with other variables as well

## Between variables

#Boxplot between svi and lcp
sns.boxplot(x='svi',y='lcp',data=data)
plt.show()
#males with higher lcp has svi 
# this can suggest that presence of svi denotes cancer

#Boxplot between gleason and age
sns.boxplot(x='gleason',y='age',data=data)
plt.show()
#gleason value 8 has neglible count,otherwise as age increases, gl score shows increase


#Boxplot between lbph and svi
sns.boxplot(x='svi',y='lbph',data=data)
plt.show()
#median value for men with no svi was higher 
#this can denote that only males with malignant cancer tissue has svi

#Scatterplot between lbph and lweight
sns.scatterplot(x='lpsa',y='lbph',data=data)
plt.show()
#lbph shows an increase with lweight 


# Scatterplot between lcavol and lweight with lpsa
sns.scatterplot(x='lcavol',y='lweight',hue ='lpsa',data=data)
plt.show()
#lpsa is increasing with cancer volume but no clear trend is seen between lpsa and lweight

# Scatterplot between age and lcavol with lpsa
sns.scatterplot(x='age',y='lcavol',hue ='lpsa',data=data)
plt.show()
#lpsa is high in older men with high cancer volume
#majority of men are between 60-70 age group

# Scatterplot between lcavol and lcp with lpsa
sns.scatterplot(x='lcp',y='lcavol',hue ='lpsa',data=data)
plt.show()
#as lcavol increases so does lpsa and lcp and vice versa
#it suggests that its highly correlated to one and other

#====================================================================================================

## Model Building

# Linear Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Model 1 - full data set

# Separating input and output features

x1 = data.drop(['lpsa'], axis=1) 
y1 = data.filter(['lpsa'], axis = 1)

# Dummy encoding the categorical data
x1 = pd.get_dummies(x1, drop_first = True) 

# Splitting data into test and train

X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.3, 
                                                    random_state = 3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)

#determining r2 value
model_r2 = r2_score(y_test, y_pred)
print(model_r2)

#RMSE value
np.sqrt(mean_squared_error(y_test, y_pred))

#determining adjusted R2 value
adj_r2 = 1 - ((1 - model_r2) * ((len(y_pred) - 1) / 
                     (len(y_pred) - len(X_test.columns) - 1)))
print(adj_r2)

# We obtain a RMSE value of 0.7914 with a R2  value of 53.83%

#------------------------------------------------------------------------------------------------

#MLR after dropping insignificant variables

#Dropped variables: lbph,age,pgg45

# Model 2

# Separating input and output features

x2 = data.drop(['lpsa','lbph','age','pgg45'], axis=1) 
y2 = data.filter(['lpsa'], axis = 1)

# Dummy encoding the categorical data
x2 = pd.get_dummies(x2, drop_first = True) 

# Splitting data into test and train

X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size = 0.3, 
                                                    random_state = 3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)

#determining r2 value
model_r2 = r2_score(y_test, y_pred)
print(model_r2)

#RMSE value
np.sqrt(mean_squared_error(y_test, y_pred))

#determining adjusted R2 value
adj_r2 = 1 - ((1 - model_r2) * ((len(y_pred) - 1) / 
                     (len(y_pred) - len(X_test.columns) - 1)))
print(adj_r2)


#We obtain a RMSE value of 0.7865 with a R2  value of 55.950%


#The multiple linear regression model built after dropping variables 
#such as pgg45, lbph & age showed higher r2 value and a lower RMSE value 
#compared to model including all the features so choosing this model 
#to predict lpsa will be appropriate.

#===============================================================================================================================



























