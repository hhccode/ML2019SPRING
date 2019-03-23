import numpy as np
from LogisticRegression import LogisticRegression

def extract_features(normalization):
    X = []
    with open("X_train", "r") as f:
        f.readline()
        for line in f:
            line = line.strip("\n").split(",")
            line = [float(num) for num in line]
            X.append(line)
    X = np.array(X)

    if normalization:
        MAX = np.amax(X, axis=0)
        MIN = np.amin(X, axis=0)

        for i in range(len(X)):
            for j in [0,1,3,4,5]:
                X[i][j] = (X[i][j] - MIN[j]) / (MAX[j] - MIN[j])

        for j in [0,1,3,4,5]:
            X = np.hstack((X, np.array([X[:,j]**2]).T))
            X = np.hstack((X, np.log([X[:, j]+1e-10]).T))

        X = np.hstack((np.ones((len(X),1)), X))

    training_X = np.array(X)

    y = []
    with open("Y_train", "r") as f:
        f.readline()
        for line in f:
            y.append(float(line.strip("\n")))
    training_Y = np.array(y)

    X = []
    with open("X_test", "r") as f:
        f.readline()
        for line in f:
            line = line.strip("\n").split(",")
            line = [float(num) for num in line]
            X.append(line)
    X = np.array(X)
    
    if normalization:
        for i in range(len(X)):
            for j in [0,1,3,4,5]:
                X[i][j] = (X[i][j] - MIN[j]) / (MAX[j] - MIN[j])

        for j in [0,1,3,4,5]:
            X = np.hstack((X, np.array([X[:,j]**2]).T))
            X = np.hstack((X, np.log([X[:, j]+1e-10]).T))

        X = np.hstack((np.ones((len(X),1)), X))
    
    testing_X = np.array(X)

    return training_X, training_Y, testing_X


def sigmoid(z):
    return 1 / (1 + np.exp(-z))  

if __name__ == "__main__":
    
    training_X, training_y, testing_X = extract_features(normalization=True)
    
    N, dimension = training_X.shape

    model = LogisticRegression(training_X, training_y)
    weight = model.GradientDescent()
    testing_y = sigmoid(np.dot(testing_X, weight))

    
    with open("output.csv", "w") as f:
        print("id,label", file=f)
        for i in range(testing_y.shape[0]):
            if testing_y[i] >= 0.5:
                print("{},1".format(i+1), file=f)
            else:
                print("{},0".format(i+1), file=f)
    