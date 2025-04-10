##################################################################################################################################
# Granny Smith Apples (GS)

##################################################################################################################################
# ___Cell no. 1___

# Python packages 
import pandas as pd # for importing data into data frame format
import seaborn as sns # For drawing useful graphs, such as bar graphs
import numpy as np
import matplotlib.pyplot as plt


import sys

sys.path.append("..")

from source.utils import split #  a pre-defined function to split the data into training and testing

# ___Cell no. 2___

%store -r X
%store -r Y
%store -r df
print(X.shape) # printing the shape the dataframe X


# ___Cell no. 3___

Y = Y.map({'S': 1, 'B': 0})
Y



# ___Cell no. 4___

Xtrain, Xtest, Ytrain, Ytest  = split(X, Y)


# ___Cell no. 5___

print(Xtrain.shape)
print(Ytrain.shape)



















