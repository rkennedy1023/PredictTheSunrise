# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:12:49 2018

@author: rkennedy
"""


print(frame.head())
print(frame.dtypes)



#Jan = frame.loc[frame['month1'] == 1]  
#Jan1 = Jan.loc[Jan['day1'] == 1]
#print(Jan1)



# split data into X and y
X = frame.iloc[:,0:2].values
Y = frame.iloc[:,2].values


#print(X)

#X.apply(pd.to_numeric, errors='ignore')
pd.to_numeric(Y, errors='coerce') # or pd.to_numeric(s, errors='raise')
#pd.to_numeric(X, errors='coerce') # or pd.to_numeric(s, errors='raise')
Y.dropna()
#X.dropna()


#Y.astype(int)

#print(X)
print(Y)

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)



#model = XGBClassifier()
#model.fit(X_train, y_train)
#print(model)
