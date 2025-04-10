##################################################################################################################################
# Royal Gala Apples (RG)

##################################################################################################################################

import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 

# ___Cell no. 2___
import os
import pandas as pd 

file_path = 'C:/Users/joshu/OneDrive/Desktop/ApplesML/data/Detect-RG.xlsx'
df = pd.read_excel(file_path) 

print(f"Successfully loaded data from: {file_path}") 



# ___Cell no. 3___
df.head(5) 

# ___Cell no. 4___
df_shape = df.shape 
print("the shape of the infrared intensity data is "+str(df_shape) ) 


print("where " + str(df_shape[0]) +" is the number of rows, and")
print(str(df_shape[1]) +" is the number of columns")





# ___Cell no. 5___
wavenumbers = np.float64(df.columns[4:])
wavelengths = (1/wavenumbers)*10**7 
print("\n Example: wave number "+str(wavenumbers[0])+" in inverse centimeters converts to a wavelength of "+ str(wavelengths[0]) + " in nanometers\n")

df.columns.values[4:] = np.round(wavelengths, 3) 

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

X = df.iloc[:, 4: ]
X.head(3)



# ___Cell no. 9___

Y = df['Condition']


# ___Cell no. 10___
np.random.seed(42)
n = 50
randIx = np.random.choice(len(df), n, replace=False)
 



# ___Cell no. 11___

Xn = X.to_numpy(dtype = 'float')
Yn = Y.to_numpy(dtype = 'str')


Xn = Xn[randIx,:]
Yn = Yn[randIx]


ns,nw = np.shape(Xn)


S_Flag = (Yn =='S')
B_Flag = (Yn == 'B')


########

plt.figure(figsize=(6, 4))


plt.plot(np.array(X.columns),np.transpose(Xn[B_Flag,:])[:,:1],'b-', label = "B") 
plt.plot(np.array(X.columns),np.transpose(Xn[B_Flag,:])[:,1:],'b-') 
    

plt.plot(np.array(X.columns),np.transpose(Xn[S_Flag,:])[:,:1],'r:', label = "S") 
plt.plot(np.array(X.columns),np.transpose(Xn[S_Flag,:])[:,1:],'r:')

plt.title("RG apples", fontweight ='bold', fontsize =12)    
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
Xn = X.to_numpy(dtype = 'float')
Yn = Y.to_numpy(dtype = 'str')


Xn = Xn[randIx,:]
Yn = Yn[randIx]


ns,nw = np.shape(Xn)


S_Flag = (Yn =='S')
B_Flag = (Yn == 'B')

#####

plt.figure(figsize=(6, 4))

plt.plot(np.array(X.columns),np.transpose(Xn[B_Flag,:])[:,:1],'b-', label = "B")
plt.plot(np.array(X.columns),np.transpose(Xn[B_Flag,:])[:,1:],'b-')
    

plt.plot(np.array(X.columns),np.transpose(Xn[S_Flag,:])[:,:1],'r:', label = "S")
plt.plot(np.array(X.columns),np.transpose(Xn[S_Flag,:])[:,1:],'r:')

plt.title("RG apples", fontweight ='bold', fontsize =12)    
plt.xlabel("Wavelength (nm)", fontweight ='bold', fontsize =12)
plt.ylabel("Absorbance (au)", fontweight ='bold', fontsize =12)
plt.ylim([-3,4])

plt.legend()

plt.show()