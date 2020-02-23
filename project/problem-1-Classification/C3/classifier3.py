import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


traindata3  = pd.read_csv('TrainData3.csv',sep=",",header=None)
label3 = pd.read_csv('TrainLabel3.csv',header=None)
testdata3  = pd.read_csv('TestData3.csv',sep=",",header=None)

traindata3 = traindata3.replace(1e99,np.nan)
testdata3 = testdata3.replace(1e99,np.nan)

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN", strategy='median', axis=0)
traindata3 = imp.fit_transform(traindata3)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(traindata3)
scaled_df = pd.DataFrame(scaled_df)


train_features, test_features, train_labels, test_labels = train_test_split(traindata3, label3, 
                                                                            test_size = 0.33, 
                                                                            random_state = 42)

from sklearn import svm
from sklearn.model_selection import GridSearchCV


parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(train_features, np.ravel(train_labels))


predictions = clf.predict(test_features)
predictions = pd.DataFrame(predictions)


from sklearn.metrics import accuracy_score
print('accuracy :' ,accuracy_score(test_labels, predictions))


from sklearn.metrics import confusion_matrix
print('confusion matrix :',confusion_matrix(test_labels, predictions))


from sklearn.metrics import classification_report
print('classification report :',classification_report(test_labels, predictions))

#without splitting the data

clfr = GridSearchCV(svc, parameters)
clfr.fit(traindata3, np.ravel(label3))


predictions = clfr.predict(testdata3)
predictions = pd.DataFrame(predictions)


np.savetxt('induriClassification3.txt', predictions, fmt='%10.5f', delimiter=',',newline='\n')
