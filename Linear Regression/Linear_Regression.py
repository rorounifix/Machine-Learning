import matplotlib.pyplot as plt
import math
from random import uniform
import csv


class LogisticRegression:

    def __init__(self):
        self.theta0 = 0
        self.theta1 = 0
        self.learning_rate = 0.0001
        self.error = 0
        self.training_data = ""
        self.data_len = 0
        self.data_x = []
        self.data_y = []
        self.iteration = 500
        

    def data_set(self,data):
        
        with open(data, 'r') as file:
            read = csv.reader(file)
            
            for data in read:
                
                if len(data) != 0:
                    self.data_x.append(float(data[0]))
                    self.data_y.append(float(data[1]))
            self.data_len += len(self.data_x)

         

    def cost_func(self):
        
        data_len = self.data_len
        data_x = self.data_x
        data_y = self.data_y
        
        err = 0
        for i in range(data_len):
            error = ((self.theta0 + (self.theta1 * data_x[i])) - data_y[i])**2
            err += (1/(2*data_len)) * error              
        self.error = err
        return err
        

    def gradient_decent(self):
        data_len = self.data_len
        data_x = self.data_x
        data_y = self.data_y
        theta0 = self.theta0
        theta1 = self.theta1
        for _ in range(self.iteration):
            error_0 = 0
            error_1 = 0
            for i in range(data_len):
                error_0 += ((theta0 + (theta1 * data_x[i])) - data_y[i])
                error_1 += ((theta0 + (theta1 * data_x[i])) - data_y[i]) * data_x[i]

            theta0 = theta0 - (self.learning_rate * ((1/data_len) * error_0))
            theta1 = theta1 - (self.learning_rate * ((1/data_len) * error_1))
        
        
        print("theta0 : {}  theta1 : {}".format(theta0,theta1))
        self.plot(theta0,theta1)
        self.theta0 = theta0
        self.theta1 = theta1
                        

    def plot_error(self):
        
        fig = plt.figure(1)
        
        ax1 = fig.add_subplot(311)
        ax1.plot([1,2,3,4], [1,10,100,1000])

        ax2 = fig.add_subplot(312)
        ax2.plot([1,2,3,4], [1,10,100,1000])

        ax3 = fig.add_subplot(313)
        ax3.plot([1,2,3,4], [1,10,100,1000])

        plt.show()





    def plot(self,theta0,theta1):
        x = self.data_x
        y = self.data_y
        
        plt.figure(1)
        plt.title("Hypothesis")
        plt.xlabel("Theta 1")
        plt.ylabel("Theta 0")
        plt.grid('on')
        plt.scatter(x,y,color="red",label="data")
        plt.plot([0,max(self.data_x)],[(theta0),(max(self.data_y) * theta1)],label="gradient")
        plt.legend(loc="upper left")


        
        
        plt.show()
        
        
        

    def forward(self):
        data = self.training_data
        self.data_set(data)
        print("Before running the data Total Error : ",self.cost_func())
        self.gradient_decent()
        print("After the gradient decent Total Error : ",self.cost_func())
##        self.plot_error()


if __name__ == "__main__":

    L = LogisticRegression()
    L.training_data = "data_.csv"
    L.forward()
