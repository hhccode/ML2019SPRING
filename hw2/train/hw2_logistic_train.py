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
        MEAN = np.mean(X, axis=0)
        STD = np.std(X, axis=0)
        
        for r in range(len(X)):
            X[r] = (X[r] - MEAN) / STD
        X = np.hstack((np.ones((len(X),1)), X))

        for r in [1,2,4,5,6]:
            X = np.hstack((X, np.array([X[:,r]**2]).T))


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
        for r in range(len(X)):
            X[r] = (X[r] - MEAN) / STD
        X = np.hstack((np.ones((len(X),1)), X))
        for r in [1,2,4,5,6]:
            X = np.hstack((X, np.array([X[:,r]**2]).T))

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
    