"""
DATA PREPROCESSING

Created on Tue JULY 10 13:48:01 2018

@author: Rivindu Wijayarathna
"""
#import Lib
import numpy as n
import matplotlib.pyplot as pl
import pandas as p

#get data file Note:: file and this sourse code must be in same folder
dataset = p.read_csv('Data.csv')

#Remove Last colume 
x = dataset.iloc[:,:-1].values
#show 4 colum index :: 0,1,2...
y = dataset.iloc[:,3].values

#Taking care of missing Data
from sklearn.preprocessing import Imputer
imputer  = Imputer(missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True)
imputer = imputer.fit(x[:, 1: 3])
x[:,1:3] = imputer.transform(x[:,1:3])

#Encoding Catogarical Data
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelEncoder_x = LabelEncoder()
x[: , 0] = labelEncoder_x.fit_transform(x[: , 0]) 
onhotencoder = OneHotEncoder(n_values=None, categorical_features=[0], categories=None, sparse=True, dtype=n.float64, handle_unknown='error')
x= onhotencoder.fit_transform(x).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y) 

#Splitting the Dataset in to the training Set AND test Set
from sklearn.model_selection import train_test_split
x_train,x_test ,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
