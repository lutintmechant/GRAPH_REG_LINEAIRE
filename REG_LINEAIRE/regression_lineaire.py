import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=100,n_features=2,centers=2,random_state=0)
y= y.reshape((y.shape[0],1))

print('dimension de X',X.shape)
print('dimension de y',y.shape)

plt.scatter(X[:,0],X[:,1],c=y, cmap='summer')
plt.show()



def initialisation(X):
    W = np.random.randn(X.shape[1],1)
    b = np.random.randn(1)
    return(W, b)

#W,b = initialisation(X)

def model(X,W,b):
    Z= X.dot(W) + b
    A = 1/(1+np.exp(-Z))
    return A
 
#A = model(X,W,b)

def log_loss(A,y):
     return 1/len(y) * np.sum(-y * np.log(A) - (1 -y) * np.log(1- A))

def gradient(A,X,y):
    dW = 1/len(y) * np.dot(X.T,A - y)
    db = 1 / len(y) * np.sum(A-y)
    return (dW,db)

#dW,db = gradient(A,X,y)

def update(dW,db,W,b,learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W,b)

def predict(X,W,b):
    A = model(X,W,b)
    return A >= 0.5
from sklearn.metrics import accuracy_score

def artificial_neuron(X,y,learning_rate = 0.1,n_iter = 100):
    # initialisation W,b
    W,b = initialisation(X)
    Loss = []
    
    for i in range(n_iter):
        A = model(X,W,b)
        dW,db = gradient(A,X,y)
        Loss.append(log_loss(A,y))
        W,b = update(dW,db,W,b,learning_rate)
    
    y_pred = predict(X,W,b)
    print(accuracy_score(y,y_pred))
    
    plt.plot(Loss)
    plt.show()
    
    return (W, b)


W,b = artificial_neuron(X,y)
new_plant = np.array([2,1])
x0 = np.linspace(-1,4,100)
x1 = (-W[0] * x0 -b )/ W[1]
plt.scatter(X[:,0],X[:,1],c=y, cmap='summer')
plt.scatter(new_plant[0],new_plant[1],c='r')
plt.plot(x0,x1, c='orange', lw=3)
plt.show()
print(predict(new_plant,W,b))