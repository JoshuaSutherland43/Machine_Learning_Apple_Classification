from sklearn.model_selection import train_test_split

def mean_df (pd):
    return pd.mean()
def sum_df (pd):
    return pd.sum()
def skew_df (pd):
    return pd.skew()
def kurt_df (pd):
    return pd.kurt()
def var_df (pd): return pd.var()

def creat_rollingData (df, window_arr = [10, 20, 30, 40, 50, 100], method = sum_df, ax = 1):
    df_arr = []
    for w in window_arr:
        df_tmp = df.copy()
        
        df_tmp =  method(df_tmp.rolling(w, axis = ax ))
        # df_tmp =  df_tmp.rolling(w, axis = ax ).mean()
        
        df_tmp = df_tmp.iloc[:, w-1::w]
        df_arr.append(df_tmp)
    return df_arr

def creat_rollingData2 (df, window_arr = [10, 20, 30, 40, 50, 100], method = var_df, ax = 1):
    df_arr = []
    for w in window_arr:
        df_tmp = df.copy()
        
        df_tmp =  method(df_tmp.rolling(w, axis = ax))  # Calculate variance
        df_tmp = df_tmp.iloc[:, w-1::w]  # Selecting every w-th column after rolling
        df_arr.append(df_tmp)
    return df_arr

def creat_rollingData3(df, window_arr = [2, 3], method = skew_df, ax = 1):
    df_arr = []
    for w in window_arr:
        df_tmp = df.copy()
        
        # Apply method (e.g., skew_df) to each rolling window
        if ax == 1:
            df_tmp = df_tmp.T.rolling(w).apply(lambda x: method(x)).T
        else:
            df_tmp = df_tmp.rolling(w).apply(lambda x: method(x))
        
        df_tmp = df_tmp.iloc[:, w-1::w]  # Select columns every w-th one
        
        print(f"Rolling window {w}: Shape of DataFrame after rolling = {df_tmp.shape}")
        
        if df_tmp.shape[0] > 0 and df_tmp.shape[1] > 0:
            df_arr.append(df_tmp)
        else:
            print(f"⚠️ Rolling window {w} resulted in an empty DataFrame.")
    
    return df_arr

def split (x, y ):
    Xtrain, Xtest, Ytrain, Ytest  = train_test_split( x, y, test_size = 0.3, random_state=3, stratify=y) # splitting the data
    return Xtrain, Xtest, Ytrain, Ytest
