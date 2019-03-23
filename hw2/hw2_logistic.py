import numpy as np
import sys

'''
Use z-normalization on all features
'''

def extract_features(normalization):
    X = []
    with open(sys.argv[1], "r") as f:
        f.readline()
        for line in f:
            line = line.strip("\n").split(",")
            line = [float(num) for num in line]
            X.append(line)

    if normalization:
        MEAN = np.load("MEAN.npy")
        STD = np.load("STD.npy")

        for r in range(len(X)):
            X[r] = (X[r] - MEAN) / STD
            X[r] = np.append(1, X[r])
            
    testing_X = np.array(X)

    return testing_X


def sigmoid(z):
    return 1 / (1 + np.exp(-z))  

if __name__ == "__main__":
    
    testing_X = extract_features(normalization=True)
    
    weight = np.load("logistic_weight.npy")

    testing_y = sigmoid(np.dot(testing_X, weight))

    with open(sys.argv[2], "w") as f:
        print("id,label", file=f)
        for i in range(testing_y.shape[0]):
            if testing_y[i] >= 0.5:
                print("{},1".format(i+1), file=f)
            else:
                print("{},0".format(i+1), file=f)
    