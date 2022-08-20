import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.0001,iters=1000):
        self.lr = lr
        self.iters = iters
        self.weights = None
        self.bias = 0

    def fit(self,x,y):
        n_samples,n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iters):
            z = np.dot(x, self.weights) + self.bias
            y_predicted = 1/(1+np.exp(-z))

            dw = (1/n_samples) * np.dot(x.T,(y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db    

    def predict(self,x):
        #y_predicted = 1/(1+np.exp(-(np.dot(x,self.weights))))
        z = np.dot(x, self.weights) + self.bias
        y_predicted = 1/(1+np.exp(-z))
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
