import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


traindata2  = pd.read_csv('TrainData.csv',sep=",",header=None)
label2 = pd.read_csv('TrainLabel2.csv',header=None)
testdata2  = pd.read_csv('TestData.csv',sep=",",header=None)


train_features, test_features, train_labels, test_labels = train_test_split(traindata2, label2, 
                                                                            test_size = 0.2, 
                                                                            random_state = 42)


from sklearn.neighbors import KNeighborsClassifier
rf = KNeighborsClassifier(n_neighbors=3)

rf.fit(train_features, np.ravel(train_labels));


predictions = rf.predict(test_features)
predictions = pd.DataFrame(predictions)


from sklearn.metrics import accuracy_score
print('accuracy' , accuracy_score(test_labels, predictions))


from sklearn.metrics import confusion_matrix
print('confucion_matrix :' ,confusion_matrix(test_labels, predictions))


from sklearn.metrics import classification_report
print('classification_report : ' ,classification_report(test_labels, predictions))

#without splitting the data

from sklearn.neighbors import KNeighborsClassifier
rfknn = KNeighborsClassifier(n_neighbors=3)


rfknn.fit(traindata2, np.ravel(label2));


predictions = rfknn.predict(testdata2)
predictions = pd.DataFrame(predictions)


np.savetxt('induriClassification2.csv', predictions, fmt='%10.5f', delimiter=',',newline='\n')
