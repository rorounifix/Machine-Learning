import numpy as np
from activation_func import *
import csv
import matplotlib.pyplot as plt



class Neural_Network:


    def __init__(self):
        self.datasets = "train.csv"
        self.layers = [0,10,5,5,1]
        self.params = None
        self.learning_rate = 0.11
        self.parameter_tuning = 0.01
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
                data_train_x.append(sets[:-2])
                data_train_y.append(sets[-2])

            train_x = np.array(data_train_x,dtype=float)
            train_y = np.array(data_train_y,dtype=float)
            train_y = train_y.reshape(train_y.shape[0],1)
            s = np.sum(train_x,axis=0)/train_x.shape[0]
            X = (train_x/s)*0.001
           
                
                
                        
        return X,train_y


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
          
                        
            loss.append(cost[0][0]) #for plot 
            itr.append(i) #for plot
        self.params = params

        ##this is not accurate prediction, i need some suggestion here
        x = np.sum(Y,axis=1)
        y = np.sum(yhat,axis=1)
        print("{}% probability".format(round((y/x)[0]*100,3)))
        self.plot(itr,loss)

    def predict(self,X):
        X = X.T
        params = self.params
        pred = self.forward(X,params)[1]
        print(round(pred[0]*100,2),"%")
        

    

if __name__ == "__main__":
    NN = Neural_Network()
    np.random.seed(1)
    learning_rate = NN.learning_rate
##    X = np.random.randn(800,5)
##    Y = np.random.randint(2,size=X.shape[0])
    X,Y = NN._datasets()
##    NN.params = NN.set_params(X,Y)[2]
    NN.L_model(X,Y,learning_rate)
##    X_test = np.array([[2,3,4,5,6],[1,2,3,1,2]])
##    NN.predict(X_test)
    

   
