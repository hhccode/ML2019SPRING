import pandas as pd
import numpy as np
import sys

ALL_FEATURES = {"AMB_TEMP": 0, "CH4": 1, "CO": 2, "NMHC": 3, "NO": 4,
                "NO2": 5, "NOx": 6, "O3": 7, "PM10": 8, "PM2.5": 9,
                "RAINFALL": 10, "RH": 11, "SO2": 12, "THC": 13, "WD_HR": 14,
                "WIND_DIREC": 15, "WIND_SPEED": 16, "WS_HR": 17}

def to_float(x):
    if x == "NR":
        return 0.0
    else:
        return float(x)

def extract_features(wanted_features, hour=9, nomorlized=True):
    # Read the testing file
    testing_df = pd.read_csv(sys.argv[1], encoding='big5', header=None).values

    # Store the testing date as the floating number
    testing_data = []
    for i in range(len(testing_df)):
        testing_data.append(list(map(to_float, testing_df[i, 2:])))
    testing_data = np.array(testing_data)

    row, col = testing_data.shape
    
    temp_X = []
    
    for i in range(0, row, 18):
        temp_X.append(testing_data[i:i+18, :].flatten())
    temp_X = np.array(temp_X)
    

    MEAN = np.load("MEAN.npy")
    STD = np.load("STD.npy")
    if nomorlized:
        for r in range(temp_X.shape[0]):
            for c in range(1, temp_X.shape[1]):
                if not STD[c] == 0 :
                    temp_X[r][c] = (temp_X[r][c]- MEAN[c]) / STD[c]

    
    testing_X = [[1] for r in range(temp_X.shape[0])]
    for r in range(temp_X.shape[0]):
        for power in range(len(wanted_features)):
            for wanted_feature in wanted_features[power]:
                testing_X[r].extend(temp_X[r, 9*ALL_FEATURES[wanted_feature]:9*(ALL_FEATURES[wanted_feature]+1)] ** (power+1))
        
    testing_X = np.array(testing_X)
    
    return testing_X
    


if __name__ == "__main__":

    features = [["CO", "NO2", "NOx", "O3", "PM10", "PM2.5", "SO2"],
                ["PM10", "PM2.5"]]
    
    testing_x = extract_features(wanted_features=features)

    weight = np.load("hw1_weight.npy")

    testing_y = np.dot(testing_x, weight)

    with open(sys.argv[2], "w") as f:
        print("id,value", file=f)
        for i in range(testing_y.shape[0]):
            print("id_{},{}".format(i, testing_y[i]), file=f)
    
    