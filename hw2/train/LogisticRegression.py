import numpy as np

class LogisticRegression():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.optimize = {"Adagrad": self._Adagrad, "Adam": self._Adam}

    def GradientDescent(self, optimizer="Adagrad", **kwargs):
        return self.optimize[optimizer](**kwargs)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  

    def _Adagrad(self, lr=1, epoch=10000, lambda_=0):
        size, dim = self.X.shape

        w = np.zeros(dim)
        w_GRAD = np.zeros(dim)

        for i in range(epoch):
            predict_y = self._sigmoid(np.dot(self.X, w))
            loss = predict_y - self.y
            
            w_grad = np.dot(self.X.T, loss) + lambda_ * 2 * w
            w_GRAD += w_grad ** 2

            w -= lr * w_grad / (np.sqrt(w_GRAD) + 0.0005)
            
            if i % 500 == 0:
                print("Iteration = {}, Lambda = {}".format(i, lambda_))
                print(self._GetAcc(predict_y, size))
                
        print("Iteration = {}, Lambda = {}".format(epoch, lambda_))
        print(self._GetAcc(predict_y, size))

        return w
        

    def _Adam(self, lr=0.0001, epoch=10000, beta_one=0.9, beta_two=0.999, epsilon=1e-8, lambda_=0):
        size, dim = self.X.shape

        w = np.zeros(dim)
        
        Vw = np.zeros(dim)   # Momentum for w
        Sw= np.zeros(dim)   # RMSprop   for w
        
        for i in range(epoch):
            predict_y = self._sigmoid(np.dot(self.X, w))
            loss = predict_y - self.y
            
            w_grad = 2 * np.dot(self.X.T, loss) + lambda_ * 2 * w

            Vw = beta_one * Vw + (1 - beta_one) * w_grad
            Sw = beta_two * Sw + (1 - beta_two) * (w_grad ** 2)

            Vw_hat = Vw / (1 - pow(beta_one, i+1))
            Sw_hat = Sw / (1 - pow(beta_two, i+1))

            w -= lr * Vw_hat / (np.sqrt(Sw_hat) + epsilon)
            
            if i % 500 == 0:
                print("Iteration=", i)
                print(self._GetAcc(predict_y, size))

        print("Iteration=", epoch)
        print(self._GetAcc(predict_y, size))

        return w

    def _GetAcc(self, prediction, data_size):
        cnt = 0
        for i in range(data_size):
            if prediction[i] >= 0.5:
                if self.y[i] == 1.0:
                    cnt += 1
            else:
                if self.y[i] == 0.0:
                    cnt += 1
        
        return cnt / data_size