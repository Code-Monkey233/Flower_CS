from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.utils import shuffle
from myconstants import *


def preprocessing(filepath:str):
    df = pd.read_csv(filepath)
    df.columns = ['Datetime', 'Occupancy']
    def generate_time_lags(df, n_lags):
        df_n = df.copy()
        for n in range(1, n_lags + 1):
            df_n[f"lag{n}"] = df_n['Occupancy'].shift(n)  #注意这里是以series的形式存储的，不读取列名
        df_n = df_n.iloc[n_lags:]
        return df_n
    df_generated = generate_time_lags(df, 5)
    df_generated['Datetime']=[pd.to_datetime(x) for x in df_generated['Datetime']]
    df_generated.set_index('Datetime',inplace=True)
    print(df_generated)
    df_features = (df_generated
               .assign(day = df_generated.index.day)
               .assign(month = df_generated.index.month)
               .assign(day_of_week = df_generated.index.dayofweek)
               .assign(week_of_year = pd.Index(df_generated.index.isocalendar().week))
               )
    def feature_label_split(df, target_col): #划分X，y
        y = df[[target_col]]
        X = df.drop(columns=[target_col])
        return X, y

    def train_val_test_split(df, target_col, test_ratio):
        X, y = feature_label_split(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
        X_train, y_train = shuffle(X_train, y_train)
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_val_test_split(df_features,'Occupancy', 0.2)
    def get_scaler(scaler):
            scalers = {
                "minmax": MinMaxScaler,
                "standard": StandardScaler,
                "maxabs": MaxAbsScaler,
                "robust": RobustScaler,
            }
            return scalers.get(scaler.lower())()
    scaler = get_scaler('minmax')
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train)
    y_test_arr = scaler.transform(y_test)
    return X_train_arr, X_test_arr, y_train_arr, y_test_arr, X_test, scaler

#preprocessing('102.csv')
def preprocessing_centralized(filepaths):
            
    # Loading in data
    df = pd.read_csv(filepaths[0])
    for i in range(1, len(filepaths)):
        df2 = pd.read_csv(filepaths[i])
        df._append(df2, ignore_index = True)
    df.columns=['Datetime', 'Occupancy']
    df.set_index('Datetime',inplace=True)
    df.index = pd.to_datetime(df.index)

    def generate_time_lags(df, n_lags):
        df_n = df.copy()
        for n in range(1, n_lags + 1):
            df_n[f"lag{n}"] = df_n["Occupancy"].shift(n)
        df_n = df_n.iloc[n_lags:]
        return df_n
        
    input_dim = N_FEATURES - 4

    df_generated = generate_time_lags(df, input_dim)

    df_features = ( df_generated
                    .assign(day = df_generated.index.day)
                    .assign(month = df_generated.index.month)
                    .assign(day_of_week = df_generated.index.dayofweek)
                    .assign(week_of_year = pd.Index(df_generated.index.isocalendar().week)))


    # Train test split

    def feature_label_split(df, target_col):
        y = df[[target_col]]
        X = df.drop(columns=[target_col])
        return X, y

    def train_val_test_split(df, target_col, test_ratio):
        X, y = feature_label_split(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
        X_train, y_train = shuffle(X_train, y_train)
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_val_test_split(df_features, 'Occupancy', 0.2)

    def get_scaler(scaler):
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()

    scaler = get_scaler('minmax')
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train)
    y_test_arr = scaler.transform(y_test)
    
    return X_train_arr, X_test_arr, y_train_arr, y_test_arr, X_test, scaler
preprocessing_centralized(['102.csv','1162.csv'])