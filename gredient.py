import numpy as np
import matplotlib.pyplot as plt

def gradient_decent(m,b,points,alpha):
    f=0
    m = 0
    b = 0
    for i in range(iteration):
        m,b = cost(m,b,points,alpha)
        print('b = {}, m = {}'.format(b,m))
        f += 1
    print(f)
    return [m,b]



def cost(m,b,points,alpha):
    points = np.array(points)
    error_m = 0
    error_b = 0
    
    N = float(len(points))
    
    for i in range(0,len(points)):
        x = points[i, 0]
        y = points[i, 1]
        
        guess = b + m * x
        error_m +=  ((1/N) * ((guess - y)) * x) * alpha
        error_b +=  ((1/N) * (guess - y))  * alpha
        
    new_b = b - (error_b )
    new_m = m - (error_m )
    
    
    
##    print(' m = ', new_m)
##    print(' b = ', new_b)
    return [new_m,new_b]

def no_error():
    pass


def run():
##    cost(m,b,points,alpha)
    final = gradient_decent(m,b,points,alpha)


if __name__ == "__main__":
##    points = np.genfromtxt('data.csv', delimiter = ",")
            #  x|y
    points = [[1,1],
              [2,2],
              [3,3]]
    
    alpha = 0.01
    m = 0
    b = 0
    iteration = 1000

    run()


