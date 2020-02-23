import numpy as np
import pandas as pd
from fancyimpute import KNN


dataset  = pd.read_csv('MissingData2.csv',sep=",",header=None)
dataset = dataset.replace(1e99,np.NaN)


dataset = dataset.values
df_filled = pd.DataFrame(KNN(3).complete(dataset))

np.savetxt('induriMissingResult2.txt', df_filled, delimiter=',',newline='\n')
