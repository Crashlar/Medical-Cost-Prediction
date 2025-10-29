import pandas as pd 
import numpy as np

def StandartScalerdata(csv_file):
    df = pd.read_csv(csv_file)
    from sklearn.preprocessing import StandardScaler

    # do standard sclar but before extract the target 
    X  = df.drop("Charges" , axis= 1 )
    y = df["Charges"]


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # print(X_scaled)
    # print(y)

    return X_scaled , y 


def train_test_split(X_scaled , y):
    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X_scaled , y , test_size= 0.2 , random_state= 13)

    # return data
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    return X_train , X_test , y_train , y_test



if __name__ == "__main__":
    X_scaled , y = StandartScalerdata("insurance.csv")
    train_test_split(X_scaled= X_scaled , y= y)











