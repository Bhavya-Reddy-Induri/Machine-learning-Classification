import numpy as np
import pandas as pd

dTrain = pd.read_csv('MultLabelTrainData.csv', header= None)
dLabel = pd.read_csv('MultLabelTrainLabel.csv', header= None)
dTest = pd.read_csv('MultLabelTestData.csv', header= None)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dTrain, dLabel,test_size = 0.1, 
                                                random_state = 42)
 

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
classifier = OneVsRestClassifier(LinearSVC(C=100.)).fit(X_train, y_train).predict(X_test)


from sklearn.metrics import hamming_loss
print('hammingloss:',hamming_loss(y_test, classifier))


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, classifier, average='micro')


np.savetxt('induriMultiLabelClassification.txt', classifier, fmt='%10.5f', delimiter=',',newline='\n')
