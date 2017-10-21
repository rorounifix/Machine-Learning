import numpy as np
from activation_func import *
import csv
import matplotlib.pyplot as plt



class Neural_Network:
    
    def __init__(self):
        self.datasets = "train.csv"
        self.layers = [0,10,10,1] 
        self.params = None
        self.learning_rate = 0.01
        self.parameter_tuning = 0.1
        self.iteration = 10000
        
    def _datasets(self):

        data = self.datasets
        with open(data) as file:
            read = csv.reader(file)
            data_train_x = []
            data_train_y = []
            for _ in read:
                sets = []
                for i in _:
                    if i == "male":
                        i = 1
                    elif i == "female":
                        i = 0
                    elif len(i) == 0:
                        i = 0
                    elif i == "S":
                        i = 1
                    elif i == "Q":
                        i = 2
                    elif i == "C":
                        i = 3
                    sets.append(i)
                data_train_x.append(sets[:-1])
                data_train_y.append(sets[-1])

            train_x = np.array(data_train_x,dtype=float)
            train_y = np.array(data_train_y,dtype=float)
            train_y = train_y.reshape(train_y.shape[0],1)
            
        return train_x,train_y


    def set_params(self,X,Y):
        X = X.reshape(X.shape[1],X.shape[0])*self.parameter_tuning
        Y = Y.reshape(1,Y.shape[0])
        L = self.layers
        params = {}
        L[0] = X.shape[0]
        for i in range(1,len(L)):
            w = np.random.randn(L[i-1],L[i])
            b = np.zeros(L[i]).reshape(L[i],1)
            params.update({"w"+str(i):w ,
                           "b"+str(i):b })
        return X,Y,params
    

    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    

    def relu(self,Z):
        A = np.maximum(0,Z)
        return A

    def forward(self,X,params):
        L = self.layers
        X = X*self.parameter_tuning
        prop = {}
        prop.update({"A0":X})
        
        for i in range(1,len(L)):
            prop.update({"Z"+str(i):np.dot(params["w"+str(i)].T,prop["A"+str(i-1)])})
            if i == len(L)-1:
                prop.update({"A"+str(i):self.sigmoid(prop["Z"+str(i)])})
            else:
                prop.update({"A"+str(i):self.relu(prop["Z"+str(i)])})
        yhat = prop["A"+str(len(L)-1)]
        return prop, yhat
    
        
    def cost(self,Y,yhat):
        A = yhat
        m = Y.shape[1]
        cost = -(1/m) * np.sum(np.multiply(Y,np.log(A)) - np.multiply((1-Y),np.log(1-A)),axis=1,keepdims=True)
        return cost

    def back_prop(self,X,Y,prop,params,yhat):
        """
        prop = contains caches from 'Z' and activation
        params = contains parameters of 'w' and 'b'
        """
        A = yhat 
        m = yhat.shape[1] # number of sets
        L = len(self.layers)-1 # number of layers
        grads = {}
        grads.update({"dA"+str(L):-(np.divide(Y,A) - np.divide((1-Y),(1-A)))})
        grads.update({"dZ"+str(L):back_sigmoid(grads["dA"+str(L)],prop["Z"+str(L)])})
        grads.update({"dW"+str(L):(1/m) * np.dot(grads["dZ"+str(L)],prop["A"+str(L-1)].T)})
        grads.update({"db"+str(L):(1/m) * np.sum(grads["dZ"+str(L)],axis=1,keepdims=True)})
        for i in reversed(range(1,L)):
            grads.update({"dA"+str(i):np.dot(params["w"+str(i+1)],grads["dZ"+str(i+1)])})
            grads.update({"dZ"+str(i):back_relu(grads["dA"+str(i)],prop["Z"+str(i)])})
            grads.update({"dW"+str(i):(1/m) * np.dot(grads["dZ"+str(i)],prop["A"+str(i-1)].T)})
            grads.update({"db"+str(i):(1/m) * np.sum(grads["dZ"+str(i)],axis=1,keepdims=True)})
        return grads

    def update_params(self,grads,params,learning_rate):
        L = len(self.layers)
        new_params = {}
        for i in range(1,L):
            new_params.update({"w"+str(i):params["w"+str(i)] - (learning_rate*grads["dW"+str(i)].T),
                        "b"+str(i):params["b"+str(i)] - (learning_rate*grads["db"+str(i)])})
        return new_params


    def plot(self,X,Y):
        plt.plot(X,Y)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.show()
    
        
    def L_model(self,X,Y,learning_rate):
        
        iteration = self.iteration
        X,Y,params = NN.set_params(X,Y)
        loss = []
        itr = []
        for i in range(iteration):
            prop,yhat = NN.forward(X,params)
            cost = NN.cost(Y,yhat)
            grads = NN.back_prop(X,Y,prop,params,yhat)
            params = NN.update_params(grads,params,learning_rate)
          
                        
            loss.append(cost[0][0]) 
            itr.append(i) 
        self.params = params 
        for i in range(len(yhat[0])):
            if yhat[0,i] >= 0.5:
                yhat[0,i] = 1
            else:
                yhat[0,i] = 0
        chance = np.sum(yhat==Y)/yhat.shape[1]
        print("Probability {}%".format(round(chance*100,2)))
        self.plot(itr,loss)

    def predict(self,X):
        X = X.T
        params = self.params
        pred = self.forward(X,params)[1]
        for i in range(len(pred[0])):
            if pred[0,i] >= 0.5:
                pred[0,i] = 1
            else:
                pred[0,i] = 0
        return pred
        

if __name__ == "__main__":
    NN = Neural_Network()
    np.random.seed(1)
    learning_rate = NN.learning_rate
    X,Y = NN._datasets()
    NN.L_model(X,Y,learning_rate)

    

   
