import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
import os
from sklearn.preprocessing import StandardScaler

# === Fixed file path ===
file_path = r'C:\Users\lab_services_student\Downloads\Apple_classifaction\data\Detect-GD.xlsx'

# === Check if file exists ===
if not os.path.exists(file_path):
    print(f"❌ File not found at: {file_path}")
    exit()

print(f"✅ Successfully located file at: {file_path}")
df = pd.read_excel(file_path)
print("✅ Successfully loaded data!")

print("\n##################################################################################################################################")
print("# Golden Delicious Apples (GD)")
print("##################################################################################################################################\n")

# Head and shape
print(df.head(5))
df_shape = df.shape 
print(f"The shape of the infrared intensity data is {df_shape}")
print(f"Where {df_shape[0]} is the number of rows, and {df_shape[1]} is the number of columns")

# Convert wavenumber to wavelength
wavenumbers = np.float64(df.columns[4:])
wavelengths = (1 / wavenumbers) * 10**7 
print(f"\nExample: Wave number {wavenumbers[0]} in inverse centimeters converts to a wavelength of {wavelengths[0]} in nanometers\n")

df.columns.values[4:] = np.round(wavelengths, 3) 

# Capitalize conditions
df['Condition'] = df['Condition'].str.upper()
ax = sns.countplot(x="Condition", data=df)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', (p.get_x()+0.25, p.get_height()+0.01))
plt.title("Count of Apple Conditions (GD)")
plt.show()

# Features and labels
X = df.iloc[:, 4:]
Y = df['Condition']

# Set random seed for reproducibility
np.random.seed(42)
n = 50
randIx = np.random.choice(len(df), n, replace=False)

Xn = X.to_numpy(dtype='float')[randIx, :]
Yn = Y.to_numpy(dtype='str')[randIx]
S_Flag = (Yn == 'S')
B_Flag = (Yn == 'B')

# Plot before scaling
plt.figure(figsize=(6, 4))
plt.plot(np.array(X.columns), np.transpose(Xn[B_Flag, :])[:, :1], 'b-', label="B")
plt.plot(np.array(X.columns), np.transpose(Xn[B_Flag, :])[:, 1:], 'b-')
plt.plot(np.array(X.columns), np.transpose(Xn[S_Flag, :])[:, :1], 'r:', label="S")
plt.plot(np.array(X.columns), np.transpose(Xn[S_Flag, :])[:, 1:], 'r:')
plt.title("GD apples", fontweight='bold', fontsize=12)    
plt.xlabel("Wavelength (nm)", fontweight='bold', fontsize=12)
plt.ylabel("Absorbance (au)", fontweight='bold', fontsize=12)
plt.ylim([-.3, 2.2])
plt.legend()
plt.show()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Plot after scaling
Xn = X.to_numpy(dtype='float')[randIx, :]
Yn = Y.to_numpy(dtype='str')[randIx]
S_Flag = (Yn == 'S')
B_Flag = (Yn == 'B')

plt.figure(figsize=(6, 4))
plt.plot(np.array(X.columns), np.transpose(Xn[B_Flag, :])[:, :1], 'b-', label="B")
plt.plot(np.array(X.columns), np.transpose(Xn[B_Flag, :])[:, 1:], 'b-')
plt.plot(np.array(X.columns), np.transpose(Xn[S_Flag, :])[:, :1], 'r:', label="S")
plt.plot(np.array(X.columns), np.transpose(Xn[S_Flag, :])[:, 1:], 'r:')
plt.title("GD apples (Scaled)", fontweight='bold', fontsize=12)    
plt.xlabel("Wavelength (nm)", fontweight='bold', fontsize=12)
plt.ylabel("Absorbance (au)", fontweight='bold', fontsize=12)
plt.ylim([-3, 4])
plt.legend()
plt.show()
