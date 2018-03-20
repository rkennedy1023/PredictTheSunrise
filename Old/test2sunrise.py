# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 08:52:08 2018

@author: rkennedy
"""

#Imports for all  libraries
import pandas as pd
import glob
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Can be toggled on and off based on what you want to display. 
pd.set_option('display.max_rows', 5000000)

def get_sec(time_str):
    h, m = time_str.split(':')
    
    return int(h) * 3600 + int(m) * 60 

#print(get_sec('23:45'))


#Load all data into dataframe from the folder
path =r'C:\Users\rkennedy\projects\Data Science\Predict the Sunrise\SunRiseSet\SunRiseSet\outbound' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)


#remove junk row
frame = frame.loc[frame['sunrise1'] != 'Obse']

frame=frame.dropna()
#Drop sunset data as it is irrelevant for our model.
frame = frame.drop(['sunset1'], axis=1)

#Clean up Sunrise to transform to 

frame['sunrise_convert']=frame['sunrise1'].apply(get_sec) #convert into minutes. split 

#print(frame)

frame = frame.drop(['sunrise1'], axis=1)
#transform to numeric and remove junk. Might not need this.
frame=frame.apply(pd.to_numeric, errors='coerce')
frame=frame.dropna()

#print(frame)
#de-bugging, check for any nulls
#print(frame[frame.isnull().any(axis=1)])

#Slice data frame into arrays to pass into the model
X = frame.iloc[:,0:2].values
Y = frame.iloc[:,2].values

#confirm data looks good
#print(Y)
#print(X)

#Declare seed / test size, and prep the data for processing by the model


train_size = int(len(X) * 0.8)
x_train, x_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = Y[0:train_size], Y[train_size:len(Y)]

print(x_train)
print(y_train)
