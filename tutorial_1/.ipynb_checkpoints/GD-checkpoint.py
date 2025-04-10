import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
import os

# === Fixed file path ===
file_path = r'C:\Users\lab_services_student\Downloads\Apple_classifaction\data\Detect-GD.xlsx'

# === Check if file exists ===
if not os.path.exists(file_path):
    print(f"❌ File not found at: {file_path}")
else:
    print(f"✅ Successfully located file at: {file_path}")
    
    # Load the Excel file
    df = pd.read_excel(file_path)
    print("✅ Successfully loaded data!")

    # === Rest of your script ===

    print("\n##################################################################################################################################")
    print("# Golden Delicious Apples (GD)")
    print("##################################################################################################################################\n")

    # Head of data
    print(df.head(5)) 

    # Shape of data
    df_shape = df.shape 
    print("The shape of the infrared intensity data is " + str(df_shape) ) 
    print("Where " + str(df_shape[0]) +" is the number of rows, and")
    print(str(df_shape[1]) +" is the number of columns")

    # Convert wavenumber to wavelength
    wavenumbers = np.float64(df.columns[4:])
    wavelengths = (1 / wavenumbers) * 10**7 
    print("\nExample: Wave number " + str(wavenumbers[0]) + " in inverse centimeters converts to a wavelength of " + str(wavelengths[0]) + " in nanometers\n")

    df.columns.values[4:] = np.round(wavelengths, 3) 

    # Plot conditions
    ax = sns.countplot(x="Condition", data=df)
    df['Condition'] = df['Condition'].str.upper()
    ax = sns.countplot(x="Condition", data=df)

    for p in ax.patches:
        ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))

    plt.show()

    # Split features and label
    X = df.iloc[:, 4:]
    Y = df['Condition']

    # Random selection
    n = 50
    randIx = np.random.choice(len(df), n, replace=False)

    Xn = X.to_numpy(dtype='float')[randIx, :]
    Yn = Y.to_numpy(dtype='str')[randIx]

    ns, nw = np.shape(Xn)

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
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(x_scaled, columns=X.columns)

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
    plt.title("GD apples", fontweight='bold', fontsize=12)    
    plt.xlabel("Wavelength (nm)", fontweight='bold', fontsize=12)
    plt.ylabel("Absorbance (au)", fontweight='bold', fontsize=12)
    plt.ylim([-3, 4])
    plt.legend()
    plt.show()
