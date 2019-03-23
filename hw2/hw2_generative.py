import numpy as np
import sys

def extract_features(normalization):
    X = []
    with open(sys.argv[1], "r") as f:
        f.readline()
        for line in f:
            line = line.strip("\n").split(",")
            line = [float(num) for num in line]
            X.append(line)
    
    if normalization:
        MEAN = np.mean(X, axis=0)
        STD = np.std(X, axis=0)

        for r in range(len(X)):
            X[r] = (X[r] - MEAN) / STD

    training_X = np.array(X)
    
    y = []
    with open(sys.argv[2], "r") as f:
        f.readline()
        for line in f:
            y.append(int(line.strip("\n")))

    training_Y = np.array(y)

    X = []
    with open(sys.argv[3], "r") as f:
        f.readline()
        for line in f:
            line = line.strip("\n").split(",")
            line = [float(num) for num in line]
            X.append(line)

    if normalization:
        for r in range(len(X)):
            X[r] = (X[r] - MEAN) / STD

    testing_X = np.array(X)
    
    return training_X, training_Y, testing_X

def sigmoid(z):
    return 1 / (1 + np.exp(-z))  

if __name__ == "__main__":
    
    training_X, training_y, testing_X = extract_features(normalization=True)
    
    N, dimension = training_X.shape

    # Count the number of data in two classes
    num_class0 = 0
    for label in training_y:
        if label == 0:
            num_class0 += 1
    num_class1 = N - num_class0
    
    # Split the training data
    class0, class1= [], []
    for i in range(N):
        if training_y[i] == 0.0:
            class0.append(training_X[i])
        else:
            class1.append(training_X[i])
    class0 = np.array(class0)
    class1 = np.array(class1)
    
    # Calculate the mean and covariance
    u_class0 = np.mean(class0, axis=0)
    u_class1 = np.mean(class1, axis=0)
    
    cov_class0 = np.zeros((dimension, dimension))
    cov_class1 = np.zeros((dimension, dimension))
    for i in range(num_class0):
        cov_class0 += np.dot(np.transpose([class0[i]-u_class0]), [class0[i]-u_class0]) / num_class0
    for i in range(num_class1):
        cov_class1  += np.dot(np.transpose([class1[i]-u_class1]), [class1[i]-u_class1]) / num_class1
    
    cov = (num_class0*cov_class0 + num_class1*cov_class1) / N
    
    w = np.dot((u_class0 - u_class1), np.linalg.inv(cov))
    b = (-0.5) * np.dot(np.dot(u_class0.T, np.linalg.inv(cov)), u_class0) + (0.5) * np.dot(np.dot(u_class1, np.linalg.inv(cov)), u_class1) + np.log(float(num_class0) / float(num_class1))
    
    testing_y = sigmoid(np.dot(testing_X, w) + b)
    
    with open(sys.argv[4], "w") as f:
        print("id,label", file=f)
        for i in range(testing_y.shape[0]):
            if testing_y[i] >= 0.5:
                print("{},0".format(i+1), file=f)
            else:
                print("{},1".format(i+1), file=f)
    
    