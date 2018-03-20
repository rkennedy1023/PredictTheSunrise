# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:52:31 2018

@author: Ryan Kennedy
@purpose: Get a rudimentary introduction to data science by trying to predict the time of the sunrise.
"""

#Imports for all  libraries
import pandas as pd
import glob
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Can be toggled on and off based on what you want to display. 
#pd.set_option('display.max_rows', 5000000)


#Load all data into dataframe from the folder
path =r'C:\Users\rkennedy\projects\Data Science\Predict the Sunrise\SunRiseSet\SunRiseSet\outbound' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)

#Drop sunset data as it is irrelevant for our model.
frame = frame.drop(['sunset1'], axis=1)
#Clean up Sunrise to transform to numeric
frame['sunrise1'] = frame.sunrise1.str.replace(':' , '') #convert into minutes. split 
#remove junk row
frame = frame.loc[frame['sunrise1'] != 'Obse']

#transform to numeric and remove junk. Might not need this.
frame=frame.apply(pd.to_numeric, errors='coerce')
frame=frame.dropna()

#de-bugging, check for any nulls
#print(frame[frame.isnull().any(axis=1)])

#Convert sunrise column to int
frame['sunrise1'].astype('int')


#Slice data frame into arrays to pass into the model
X = frame.iloc[:,0:2].values
Y = frame.iloc[:,2].values

#confirm data looks good
#print(Y)
#print(X)

#Declare seed / test size, and prep the data for processing by the model
seed = 7 #if you let the random number generate random numbers, solution can change.  Specify seed at first, then you can unspecify
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
#67% will be training from any one of days in original data set.  Won't use the last 2 months 
#need to take FIRST 67%, because this is time series.  time is important.  mix and match the time periods IE year 1 and 2 for train, year 3 for test


##Train the model! 
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)

#To test the model on a smaller scale for an individual day / month
smalltest=[1,1]
y_pred_small=model.predict(smalltest)
y_pred_small=[round(value) for value in y_pred_small]
print(y_pred_small)

# make predictions for test data that we actually split out
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
print(predictions)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Figure out how far off my incorrect prediction were. rmse, residual some of all squares, rsf. 
#plot to show actual VS predicted sunrise VS time
#