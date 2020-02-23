# Import the libraries
import numpy as np
import pandas as pd

# Import Daa
dataset  = pd.read_csv('MissingData1.csv',sep=",",header=None)
dataset = dataset.replace(1e99,np.NaN)

#MICE - Multiple Imputation by Chained Equations
from fancyimpute import MICE
solver=MICE()
Imputed_dataframe=solver.complete(dataset)

#write to output file
np.savetxt('induriMissingResult1.txt', Imputed_dataframe, delimiter=',',newline='\n')
