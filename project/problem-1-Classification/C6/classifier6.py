import pandas as pd
import numpy as np


traindata6  = pd.read_csv('TrainData6.csv',sep=",",header=None)
label6 = pd.read_csv('TrainLabel6.csv',header=None)
testdata6  = pd.read_csv('TestData6.csv',sep=",",header=None)


X_train = traindata6[:488]
X_test = traindata6[489:]


y_train = label6[:488]
y_test = label6[489:]


from sklearn import linear_model
regr = linear_model.LinearRegression()


regr.fit(X_train, y_train)
print('regression_score :',regr.score(X_train, y_train))



y_pred = regr.predict(X_test)
residuals = y_test - y_pred


print("Training set score: {:.2f}".format(regr.score(X_train, y_train)))
print("Test set score: {:.7f}".format(regr.score(X_test, y_test)))


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

from sklearn.metrics import mean_squared_error, r2_score
print('Coefficients: \n', regr.coef_)

print("Mean squared error: %.2f", mean_squared_error(y_test, y_pred))

print('Variance score: %.2f' % r2_score(y_test, y_pred))

#without splitting the data
regL = linear_model.LinearRegression()
regL.fit(traindata6, label6)
regL.score(traindata6, label6)


y_pred = regL.predict(testdata6)

np.savetxt('induriClassification6.txt', y_pred, delimiter=',',newline='\n')
