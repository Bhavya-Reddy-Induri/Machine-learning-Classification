import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


traindata1  = pd.read_csv('TrainData1.csv',sep=",",header=None)
label1 = pd.read_csv('TrainLabel1.csv',header=None)
testdata1  = pd.read_csv('TestData1.csv',sep=",",header=None)


traindata1 = traindata1.replace(1e99,0)
testdata1 = testdata1.replace(1e99,0)


train_features, test_features, train_labels, test_labels = train_test_split(traindata1, label1, 
                                                                            test_size = 0.3, 
                                                                            random_state = 42)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain = sc.fit_transform(train_features)
Xtest = sc.transform(test_features)


from sklearn.decomposition import PCA
pca = PCA(n_components = 40)
Xtrain = pca.fit_transform(Xtrain)
Xtest = pca.transform(Xtest)
explained_variance = pca.explained_variance_ratio_


from sklearn import svm
rf = svm.SVC(kernel='linear', C=1, gamma=1)


rf.fit(Xtrain, np.ravel(train_labels));


predictions = rf.predict(Xtest)
predictions = pd.DataFrame(predictions)


from sklearn.metrics import accuracy_score
print('accuracy _score :' ,accuracy_score(test_labels, predictions))


from sklearn.metrics import confusion_matrix
print('confusion matrix :',confusion_matrix(test_labels, predictions))


from sklearn.metrics import classification_report
print('classification report : ',classification_report(test_labels, predictions))


#without splitting the data

Xtraindata = sc.fit_transform(traindata1)
Xtestdata = sc.transform(testdata1)


Xtraindata = pca.fit_transform(Xtraindata) 
Xtestdata = pca.transform(Xtestdata) 
explained_variance = pca.explained_variance_ratio_

from sklearn import svm
rf1 = svm.SVC(kernel='linear', C=1, gamma=1)


rf1.fit(Xtraindata, np.ravel(label1));


predictions = rf1.predict(Xtestdata)
predictions = pd.DataFrame(predictions)

#write to output file
np.savetxt('induriClassification1.txt', predictions, fmt='%10.5f', delimiter=',',newline='\n')
