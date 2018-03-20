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
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

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

#print(x_train)
#print(y_train)

'''
from pandas import Series
from matplotlib import pyplot

print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))

'''

#67% will be training from any one of days in original data set.  Won't use the last 2 months 
#need to take FIRST 67%, because this is time series.  time is important.  mix and match the time periods IE year 1 and 2 for train, year 3 for test

##Train the model! 
model = XGBClassifier()
model.fit(x_train, y_train)
print(model)

'''
#To test the model on a smaller scale for an individual day / month
smalltest=[1,1]
y_pred_small=model.predict(smalltest)
y_pred_small=[round(value) for value in y_pred_small]
print(y_pred_small)

'''
# make predictions for test data that we actually split out
y_pred = model.predict(x_test)
#predictions = [round(value) for value in y_pred]
#print(predictions)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(mean_squared_error(y_test, y_pred))
#Figure out how far off my incorrect prediction were. rmse, residual some of all squares, rsf. 
#plot to show actual VS predicted sunrise VS time
#
'''
df2 = pd.DataFrame()




def compare_arrays(A, B):
    ret = np.equal(A, B).all(axis=1)
    return ret

comp = compare_arrays(y_pred,y_test)

df2 = pd.DataFrame({'x':y_pred, 'y':y_test, 'z':comp})

print(df2)

df2.plot('x', 'y', kind='scatter')
plt.show()

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
'''