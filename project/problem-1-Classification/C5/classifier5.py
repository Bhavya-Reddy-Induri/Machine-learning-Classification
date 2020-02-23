import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#reading the dataset
traindata5  = pd.read_csv('TrainData5.csv',sep=",",header=None)
label5 = pd.read_csv('TrainLabel5.csv',header=None)
testdata5  = pd.read_csv('TestData5.csv',sep=",",header=None)

train_features, test_features, train_labels, test_labels = train_test_split(traindata5, label5, 
                                                                            test_size = 0.2, 
                                                                            random_state = 42)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42)

rf.fit(train_features, np.ravel(train_labels));


predictions = rf.predict(test_features)
predictions = pd.DataFrame(predictions)


from sklearn.metrics import accuracy_score
print('accuracy_score :' ,accuracy_score(test_labels, predictions))


from sklearn.metrics import confusion_matrix
print('confusion_matrix: ',confusion_matrix(test_labels, predictions))


from sklearn.metrics import classification_report
print('classification_report :' ,classification_report(test_labels, predictions))

#without splitting the data

rForest = RandomForestClassifier(random_state = 42)


rForest.fit(traindata5, np.ravel(label5));


predictions = rForest.predict(testdata5)
predictions = pd.DataFrame(predictions)


np.savetxt('induriClassification5.txt', predictions, fmt='%10.5f', delimiter=',',newline='\n')
