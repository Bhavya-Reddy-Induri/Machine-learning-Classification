import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#reading the dataset
traindata4  = pd.read_csv('TrainData4.csv',sep=",",header=None)
label4 = pd.read_csv('TrainLabel4.csv',header=None)
testdata4  = pd.read_csv('TestData4.csv',sep=",",header=None)
#splitting the data into test and train datasets
train_features, test_features, train_labels, test_labels = train_test_split(traindata4, label4, 
                                                                            test_size = 0.2, 
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


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42)

rf.fit(Xtrain, np.ravel(train_labels));


predictions = rf.predict(Xtest)
predictions = pd.DataFrame(predictions)

from sklearn.metrics import accuracy_score
print('accuracy_score :',accuracy_score(test_labels, predictions))


from sklearn.metrics import confusion_matrix
print('confusion_matrix :',confusion_matrix(test_labels, predictions))


from sklearn.metrics import classification_report
print(classification_report(test_labels, predictions))

#without splitting the data
Xtraindata = sc.fit_transform(traindata4)
Xtestdata = sc.transform(testdata4)

Xtraindata = pca.fit_transform(Xtraindata) 
Xtestdata = pca.transform(Xtestdata) 
explained_variance = pca.explained_variance_ratio_

rForest = RandomForestClassifier(random_state = 42)


rForest.fit(Xtraindata, np.ravel(label4));

# Use the forest's predict method on the test data
predictions = rForest.predict(Xtestdata)
predictions = pd.DataFrame(predictions)

#write to output file
np.savetxt('induriClassification4.txt', predictions, fmt='%10.5f', delimiter=',',newline='\n')

