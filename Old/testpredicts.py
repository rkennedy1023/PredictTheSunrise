# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:40:06 2018

@author: rkennedy
"""

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
pd.set_option('display.max_rows', 5000000)

def get_sec(time_str):
    h, m = time_str.split(':')
    
    return int(h) * 3600 + int(m) * 60 

print(get_sec('23:45'))

'''
#Load all data into dataframe from the folder
path =r'C:\Users\rkennedy\projects\Data Science\Predict the Sunrise\SunRiseSet\SunRiseSet\outbound' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)

#print(frame)


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
#X = frame.values
#print(X)
#confirm data looks good
#print(Y)
#print(X)

#train_size = int(len(X) * 0.66)
#train, test = X[0:train_size], X[train_size:len(X)]
#print('Observations: %d' % (len(X)))
#print('Training Observations: %d' % (len(train)))
#print('Testing Observations: %d' % (len(test)))




#Declare seed / test size, and prep the data for processing by the model
n = frame.shape[0]
print(n)


train_size = 0.7

#features_dataframe = features_dataframe.sort_values('date')
train_dataframe = frame.iloc[:int(n * train_size)]
print(train_dataframe)


test_dataframe = frame.iloc[int(n * train_size):]
print(test_dataframe.head())


print(len(train_dataframe))
print(len(test_dataframe))

X_train = train_dataframe.iloc[:,0:2].values
Y_train = train_dataframe.iloc[:,2].values

X_test = test_dataframe.iloc[:,0:2].values
Y_test = test_dataframe.iloc[:,2].values


#train_size = int(len(X) * 0.66)
#train, test = X[0:train_size], X[train_size:len(X)]

#print(train)
#print(test)





from pandas import Series
from matplotlib import pyplot

print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))


pyplot.plot(train)
pyplot.plot([None for i in train] + [x for x in test])
pyplot.show()

#67% will be training from any one of days in original data set.  Won't use the last 2 months 
#need to take FIRST 67%, because this is time series.  time is important.  mix and match the time periods IE year 1 and 2 for train, year 3 for test


##Train the model! 
model = XGBClassifier()
model.fit(X_train, Y_train)
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
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Figure out how far off my incorrect prediction were. rmse, residual some of all squares, rsf. 
#plot to show actual VS predicted sunrise VS time
#
'''