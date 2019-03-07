import pandas as pd
import numpy as np
from LinearRegression import LinearRegression


ALL_FEATURES = {"AMB_TEMP": 0, "CH4": 1, "CO": 2, "NMHC": 3, "NO": 4,
                "NO2": 5, "NOx": 6, "O3": 7, "PM10": 8, "PM2.5": 9,
                "RAINFALL": 10, "RH": 11, "SO2": 12, "THC": 13, "WD_HR": 14,
                "WIND_DIREC": 15, "WIND_SPEED": 16, "WS_HR": 17}

def to_float(x):
    if x == "NR":
        return 0.0
    else:
        return float(x)

def find_dirty(array):
    for i in range(len(array)):
        if array[i] < 0.0:
            return True

    return False

def extract_features(wanted_features, hour=9, nomorlized=True):
    # Read the training file
    training_df = pd.read_csv('train.csv', encoding='big5').values
    
    # Declare the 18-dim vector & extract 162 features
    training_data = [[] for _ in range(18)]
    for i in range(len(training_df)):
        training_data[i%18].extend(list(map(to_float, training_df[i][3:])))
    training_data = np.array(training_data)
    
    row, col = training_data.shape
    temp_X, training_y = [], []       
    
    cnt = 0
    i = 0
    while i < col-hour:
        X = training_data[:, i:i+hour].flatten()
        y = training_data[9, i+hour]
        
        temp_X.append(X)
        training_y.append(y)
        i += 1

    temp_X = np.array(temp_X)
    training_y= np.array(training_y)
    
    # Normalize the data
    if nomorlized:
        MEAN = np.mean(temp_X, axis=0)
        STD = np.std(temp_X, axis=0)
        for r in range(temp_X.shape[0]):
            for c in range(temp_X.shape[1]):
                if not STD[c] == 0 :
                    temp_X[r][c] = (temp_X[r][c]- MEAN[c]) / STD[c]
    
    training_X = [[] for _ in range(temp_X.shape[0])]
    for r in range(temp_X.shape[0]):
        for power in range(len(wanted_features)):
            for wanted_feature in wanted_features[power]:
                training_X[r].extend(temp_X[r, hour*ALL_FEATURES[wanted_feature]:hour*(ALL_FEATURES[wanted_feature]+1)] ** (power+1))
        
    training_X = np.array(training_X)

    # Read the testing file
    testing_df = pd.read_csv('test.csv', encoding='big5', header=None).values

    # Store the testing date as the floating number
    testing_data = []
    for i in range(len(testing_df)):
        testing_data.append(list(map(to_float, testing_df[i, 2+(9-hour):])))
    testing_data = np.array(testing_data)

    row, col = testing_data.shape
    
    temp_X = []
    
    for i in range(0, row, 18):
        temp_X.append(testing_data[i:i+18, :].flatten())
    temp_X = np.array(temp_X)
    
    if nomorlized:
        for r in range(temp_X.shape[0]):
            for c in range(temp_X.shape[1]):
                if not STD[c] == 0 :
                    temp_X[r][c] = (temp_X[r][c]- MEAN[c]) / STD[c]

    testing_X = [[1] for r in range(temp_X.shape[0])]
    for r in range(temp_X.shape[0]):
        for power in range(len(wanted_features)):
            for wanted_feature in wanted_features[power]:
                testing_X[r].extend(temp_X[r, hour*ALL_FEATURES[wanted_feature]:hour*(ALL_FEATURES[wanted_feature]+1)] ** (power+1))
        
    testing_X = np.array(testing_X)
    
    return training_X, training_y, testing_X
    


if __name__ == "__main__":
    
    '''
    features = [["AMB_TEMP", "CH4", "CO", "NMHC", "NO",
                "NO2", "NOx", "O3", "PM10", "PM2.5",
                "RAINFALL", "RH", "SO2", "THC", "WD_HR",
                "WIND_DIREC", "WIND_SPEED", "WS_HR"]]
    '''
    
    features = [["PM2.5"]]
    
    training_x, training_y, testing_x = extract_features(wanted_features=features, hour=9, nomorlized=True)
    
    model = LinearRegression(training_x, training_y)
    
    weight = model.GradientDescent(lr=0.01, epoch=20000, optimizer="Adam", lambda_=0.0001)

    testing_y = np.dot(testing_x, weight)

    with open("output.csv", "w") as f:
        print("id,value", file=f)
        for i in range(testing_y.shape[0]):
            print("id_{},{}".format(i, testing_y[i]), file=f)
