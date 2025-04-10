##################################################################################################################################
# Granny Smith Apples (GS)

##################################################################################################################################

import pandas as pd # for importing data into data frame format
import seaborn as sns # For drawing useful graphs, such as bar graphs
import matplotlib.pyplot as plt # This displays graphs once they have been created
import numpy as np # For handling N-DIMENSIONAL ARRAYS


# ___Cell no. 2___
import os
import pandas as pd 

file_path = 'C:/Users/joshu/OneDrive/Desktop/ApplesML/data/Detect-GD.xlsx'
df = pd.read_excel(file_path) 

print(f"Successfully loaded data from: {file_path}") 



# ___Cell no. 3___
df.head(5) # shows the first 5 rows of the data frame


# ___Cell no. 4___
df_shape = df.shape # "df.shape" produces a tuple of 2 numbers 
print("the shape of the infrared intensity data is "+str(df_shape) ) 

# The individual numbers in the tuple are accessed as follows:
print("where " + str(df_shape[0]) +" is the number of rows, and")
print(str(df_shape[1]) +" is the number of columns")





# ___Cell no. 5___
wavenumbers = np.float64(df.columns[4:])
wavelengths = (1/wavenumbers)*10**7 # changing the wavenumber to a wave length
print("\n Example: wave number "+str(wavenumbers[0])+" in inverse centimeters converts to a wavelength of "+ str(wavelengths[0]) + " in nanometers\n")

df.columns.values[4:] = np.round(wavelengths, 3) # getting just up to 3 decimal numbers
# Print first few rows
df.head(4)




# ___Cell no. 6___
ax = sns.countplot(x="Condition",data=df)


# ___Cell no. 7___
df['Condition'] = df['Condition'].str.upper()
ax = sns.countplot(x="Condition",data=df)

for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))

plt.show()



# ___Cell no. 8___
#Inputs (which is the infrared spectral data)
X = df.iloc[:, 4: ]
X.head(3)



# ___Cell no. 9___
#outputs (Sound and Bruised)
Y = df['Condition']


# ___Cell no. 10___
n = 50
randIx  = np.random.choice(len(df), n, replace=False)# Random sample without replacement (avoids duplicates)
randIx # those are the indices of randomly selected 50 apple samples



# ___Cell no. 11___
# Convert to numpy
Xn = X.to_numpy(dtype = 'float')
Yn = Y.to_numpy(dtype = 'str')

# Select only the ones to display
Xn = Xn[randIx,:]
Yn = Yn[randIx]

# number of samples, number of wavelengths
ns,nw = np.shape(Xn)

# Select Sound and Bruised samples
S_Flag = (Yn =='S')
B_Flag = (Yn == 'B')


########

plt.figure(figsize=(6, 4))

# Since we are plotting a 2D numpy array, we will need to be carful with the labels, as we will need just one label to present the type of graph (S, B) 

plt.plot(np.array(X.columns),np.transpose(Xn[B_Flag,:])[:,:1],'b-', label = "B") # just graph the first wavelength of type 'B' with the lables 
plt.plot(np.array(X.columns),np.transpose(Xn[B_Flag,:])[:,1:],'b-') # graphs the rest of the wavelengths of type 'B' without thier labels 
    
# We make the second curve dashed so that it doesn't cover up the first
plt.plot(np.array(X.columns),np.transpose(Xn[S_Flag,:])[:,:1],'r:', label = "S")  # just graph the first wavelength of type 'S' without the lables
plt.plot(np.array(X.columns),np.transpose(Xn[S_Flag,:])[:,1:],'r:') # graphs the rest of the wavelengths of type 'S' without thier labels

plt.title("GS apples", fontweight ='bold', fontsize =12)    
plt.xlabel("Wavelength (nm)", fontweight ='bold', fontsize =12)
plt.ylabel("Absorbance (au)", fontweight ='bold', fontsize =12)
plt.ylim([-.3,2.2])

plt.legend()

plt.show()


# ___Cell no. 12___
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

X = pd.DataFrame(x_scaled, columns = X.columns)

X


# ___Cell no. 13___
# Convert to numpy
Xn = X.to_numpy(dtype = 'float')
Yn = Y.to_numpy(dtype = 'str')

# Select only the ones to display
Xn = Xn[randIx,:]
Yn = Yn[randIx]

# number of samples, number of wavelengths
ns,nw = np.shape(Xn)

# Select Sound and Bruised samples
S_Flag = (Yn =='S')
B_Flag = (Yn == 'B')

#####

plt.figure(figsize=(6, 4))

plt.plot(np.array(X.columns),np.transpose(Xn[B_Flag,:])[:,:1],'b-', label = "B")
plt.plot(np.array(X.columns),np.transpose(Xn[B_Flag,:])[:,1:],'b-')
    
# We make the second curve dashed so that it doesn't cover up the first
plt.plot(np.array(X.columns),np.transpose(Xn[S_Flag,:])[:,:1],'r:', label = "S")
plt.plot(np.array(X.columns),np.transpose(Xn[S_Flag,:])[:,1:],'r:')

plt.title("GS apples", fontweight ='bold', fontsize =12)    
plt.xlabel("Wavelength (nm)", fontweight ='bold', fontsize =12)
plt.ylabel("Absorbance (au)", fontweight ='bold', fontsize =12)
plt.ylim([-3,4])

plt.legend()

plt.show()























