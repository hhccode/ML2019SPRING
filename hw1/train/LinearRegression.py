import numpy as np

class LinearRegression():
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.optimize = {"Adagrad": self._Adagrad, "Adam": self._Adam, "ClosedForm": self._ClosedForm, "AdamAndDraw": self._AdamAndDraw,}

    def GradientDescent(self, optimizer="ClosedForm", **kwargs):
        return self.optimize[optimizer](**kwargs)

    def _ClosedForm(self):
        size, dim = self.X.shape
        
        # Add the bias term
        X = []
        for r in range(size):
            X.append(np.append([1], self.X[r]))
        X = np.array(X)
        
        # Closed form solution
        from numpy.linalg import pinv
        w = np.matmul(pinv(X), self.y)
        
        predict_y = np.dot(X, w)
        loss = predict_y - self.y
        
        print("Iteration: 1")
        print(self._GetRMSE(loss, size))

        return w
    
    def _Adagrad(self, lr=1000, epoch=100000, lambda_=0):
        size, dim = self.X.shape
        print(lambda_)
        b = 0.0
        w = np.zeros(dim)
        b_GRAD = 0.0
        w_GRAD = np.zeros(dim)

        for i in range(epoch):
            predict_y = np.dot(self.X, w) + b
            loss = predict_y - self.y

            b_grad = 2 * np.sum(loss)
            w_grad = 2 * np.dot(self.X.T, loss) + lambda_ * 2 * w
            
            b_GRAD += b_grad ** 2
            w_GRAD += w_grad ** 2
        
            b -= lr * b_grad / (np.sqrt(b_GRAD) + 0.0005)
            w -= lr * w_grad / (np.sqrt(w_GRAD) + 0.0005)

            if i % 500 == 0:
                print("Iteration=", i)
                print(self._GetRMSE(loss, size))
                
        print("Iteration=", epoch)
        print(self._GetRMSE(loss, size))

        return np.append(b, w)
        

    def _Adam(self, lr=0.0001, epoch=10000, beta_one=0.9, beta_two=0.999, epsilon=1e-8, lambda_=0):
        size, dim = self.X.shape

        b = 0.0
        w = np.zeros(dim)
        
        Vb = 0.0    # Momentum for b
        Sb = 0.0    # RMSprop for b
        Vw = np.zeros(dim)   # Momentum for w
        Sw= np.zeros(dim)   # RMSprop   for w
        
        for i in range(epoch):
            predict_y = np.dot(self.X, w) + b
            loss = predict_y - self.y
            
            b_grad = 2 * np.sum(loss)
            w_grad = 2 * np.dot(self.X.T, loss) + lambda_ * 2 * w

            Vb = beta_one * Vb + (1 - beta_one) * b_grad
            Sb = beta_two * Sb + (1 - beta_two) * (b_grad ** 2)
            Vw = beta_one * Vw + (1 - beta_one) * w_grad
            Sw = beta_two * Sw + (1 - beta_two) * (w_grad ** 2)

            Vb_hat = Vb / (1 - pow(beta_one, i+1))
            Sb_hat = Sb / (1 - pow(beta_two, i+1))
            Vw_hat = Vw / (1 - pow(beta_one, i+1))
            Sw_hat = Sw / (1 - pow(beta_two, i+1))

            b -= lr * Vb_hat / (np.sqrt(Sb_hat) + epsilon)
            w -= lr * Vw_hat / (np.sqrt(Sw_hat) + epsilon)
            
            if i % 500 == 0:
                print("Iteration=", i)
                print(self._GetRMSE(loss, size))

        print("Iteration=", epoch)
        print(self._GetRMSE(loss, size))

        return np.append(b, w)

    def _AdamAndDraw(self, lr=0.0001, epoch=10000, beta_one=0.9, beta_two=0.999, epsilon=1e-8, lambda_=0):
        size, dim = self.X.shape

        b = 0.0
        w = np.zeros(dim)
        
        Vb = 0.0    # Momentum for b
        Sb = 0.0    # RMSprop for b
        Vw = np.zeros(dim)   # Momentum for w
        Sw= np.zeros(dim)   # RMSprop   for w
        RMSE = []

        for i in range(epoch):
            predict_y = np.dot(self.X, w) + b
            loss = predict_y - self.y
            
            b_grad = 2 * np.sum(loss)
            w_grad = 2 * np.dot(self.X.T, loss) + lambda_ * 2 * w

            Vb = beta_one * Vb + (1 - beta_one) * b_grad
            Sb = beta_two * Sb + (1 - beta_two) * (b_grad ** 2)
            Vw = beta_one * Vw + (1 - beta_one) * w_grad
            Sw = beta_two * Sw + (1 - beta_two) * (w_grad ** 2)

            Vb_hat = Vb / (1 - pow(beta_one, i+1))
            Sb_hat = Sb / (1 - pow(beta_two, i+1))
            Vw_hat = Vw / (1 - pow(beta_one, i+1))
            Sw_hat = Sw / (1 - pow(beta_two, i+1))

            b -= lr * Vb_hat / (np.sqrt(Sb_hat) + epsilon)
            w -= lr * Vw_hat / (np.sqrt(Sw_hat) + epsilon)
            
            RMSE.append(self._GetRMSE(loss, size))

        return RMSE

    def _GetRMSE(self, loss, data_size):
        return np.sqrt(np.sum(loss ** 2) / data_size)