'''Python 3
rorounifix@gmail.com
'''
##import numpy as np
import matplotlib.pyplot as plt


def gradient_decent(m,b,points,alpha):
    f=0
    m = 0
    b = 0
    for i in range(iteration):
        m,b = cost(round(m,10),round(b,10),points,alpha)

        f += 1
##    print('b = {}, m = {}, No of iter {}'.format(b,m,f))
    
    return [m,b]



def cost(m,b,points,alpha):
##    points = np.array(points)
    error_m = 0
    error_b = 0
    
    N = float(len(points))
    
    for i in range(0,len(points)):
        x = points[i][0]
        y = points[i][1]
        
        guess = b + m * x
        error_m +=  ((guess - y) * x) * alpha
        error_b += (guess - y)  * alpha
        
    new_b = b - (error_b ) * (1/N)
    new_m = m - (error_m ) * (1/N)

    return (round(new_m,10),round(new_b,10))


def plot(x,y):

    
    plt.scatter(x,y)
    
    plt.show()


def i():

    with open('data.csv','r+') as points:
        datas = points.readlines()
        points = []
        for data in datas:
            
            convert = data.strip('\n').split(',')
            x = float(convert[0])
            y = float(convert[1])
            points.append([x,y])
           
        return points


def compute_error(b,m,points):
    error = 0
    
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        guess = b + m * x
        error += guess - y
    return error

        
def run():
    error = compute_error(b,m,points)
    print('No. of iter = ', iteration)
    print('Before the regression m = {}, b = {}, error = {} '.format(m,b,error))
    final = gradient_decent(m,b,points,alpha)
    new_m = final[0]
    new_b = final[1]
    error = compute_error(final[1],final[0],points)
    print('After the regression m = {}, b = {}, error = {} '.format(new_m,new_b,round(error,10)))

if __name__ == "__main__":

            ##  x|y
##    points = [[0,6.4],
##              [1,8.6],
##              [2,11.5],
##              [3,17.8],
##              [4,18.4]]
    ##the answer must be b=5.9, m = 3.32
    ## y = mx+b (b = y-intercept) (m = slope)

##    points = [[1,1],
##              [2,2],
##              [3,3],
##              [4,4]]
##    
    points = i()           
    alpha = 0.0003
    m = 0
    b = 0
    iteration = 1000
    
    
    
    run()
##    plot(i()[0],i()[1])

