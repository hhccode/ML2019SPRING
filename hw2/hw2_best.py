import numpy as np
import sys

'''
Use min-max normalization on all features
Take extra log features
'''

def extract_features(normalization):
    X = []
    with open(sys.argv[1], "r") as f:
        f.readline()
        for line in f:
            line = line.strip("\n").split(",")
            line = [float(num) for num in line]
            X.append(line)
    X = np.array(X)
    
    if normalization:
        
        MAX = np.load("MAX.npy")
        MIN = np.load("MIN.npy")

        for i in range(len(X)):
            for j in [0,1,3,4,5]:
                X[i][j] = (X[i][j] - MIN[j]) / (MAX[j] - MIN[j])

        for j in [0,1,3,4,5]:
            X = np.hstack((X, np.array([X[:,j]**2]).T))
            X = np.hstack((X, np.log([X[:, j]+1e-10]).T))

        X = np.hstack((np.ones((len(X),1)), X))
    
    testing_X = np.array(X)

    return testing_X


def sigmoid(z):
    return 1 / (1 + np.exp(-z))  

if __name__ == "__main__":
    
    testing_X = extract_features(normalization=True)

    weight = np.load("best_weight.npy")

    testing_y = sigmoid(np.dot(testing_X, weight))

    with open(sys.argv[2], "w") as f:
        print("id,label", file=f)
        for i in range(testing_y.shape[0]):
            if testing_y[i] >= 0.5:
                print("{},1".format(i+1), file=f)
            else:
                print("{},0".format(i+1), file=f)
    