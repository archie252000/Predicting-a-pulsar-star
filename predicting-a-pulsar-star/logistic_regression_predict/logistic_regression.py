import numpy as np
import matplotlib.pyplot as plt
import pickle

def data_prep():
    #Loading the data of pulsar stars as numpy array
    data = np.genfromtxt('pulsar_stars.csv', delimiter = ',', skip_header=1)
    
    #m is the number of examples(number of rows)
    m = data.shape[0]
    
    #n is 20% of m, as 80% of the data is used for training and 20% for testing(testing data used in predict.py)
    n = int(m  * 0.2)
    
    #Initializing training data
    train_data = data[0:m-n]
    
    return train_data


def sigmoid(z):
    #The sigmoid function - 1 / ( 1 + e ^ -z )
    return 1/(1+np.exp(-1 * z))


def initialize_params(n):
    #Initializing parameters
    
    #Initializing the weights as n zeros, where n is the number of features
    W = np.zeros(shape =(n,1))
    
    #Initializing the bias unit
    b = 0
    
    #Storing the weights and bias units in a dictonary
    params = {"W":W, "b":b}
   
    return params


def initialize_hyper_params(n_iter = 1500,learning_rate = 0.1):
    
    #Initializing Hyper-parameters(number of iterations and learning rate)
    
    #Storing Hyper-parameters in a dictonary
    h_params = {"n_iter":n_iter, "learning_rate":learning_rate}
    
    return h_params


def compute_cost(X,y,W,b):
    #Computing cost using log error function
    
    #Our current predicted value( 1 - pulsar_star / 0 - not_a_pulsar_star )
    y_hat = sigmoid(np.dot(X,W) + b)
    m = X.shape[0]
    
    #Calculating and returning Loss
    return (-1/m) * np.sum((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)),keepdims=True)


def gradient(W,b,X,y):
    #calculating gradient
    
    #hypothesis(Predicted value)
    y_hat = sigmoid(np.dot(X,W) + b) 
    # dJ/dW where J is log error function
    dW = np.dot(X.T,(y_hat - y))
    # dJ/db
    db = np.sum(y_hat - y, keepdims=True)
    grads  = {"dW":dW, "db":db}
    return grads


def graident_descent(X,y,params,h_params):
    #Performing gradient descent
    
    #Getting parameters and hyper-parameters
    n_iter = h_params["n_iter"]
    learning_rate = h_params["learning_rate"]
    W = params["W"]
    b = params["b"]
    m = X.shape[0]
    #List that holds cost at the end of every iteration of gradient descent 
    all_costs = []
    iter = [i for i in range(n_iter)]
    for j in range(n_iter):
        grads = gradient(W,b,X,y)
        W = W - (learning_rate/m)*grads["dW"]
        b = b - (learning_rate/m)*grads["db"]
        cost  = compute_cost(X, y, W,b)
        if j % 100 == 0:
         print(cost)
        all_costs.append(float(cost))
    
    #Plotting cost vs iteration graph
    plt.plot(all_costs,iter)
    plt.show()
    return {"W":W, "b":b}
if __name__ == "__main__":
                
    
    #Training the model
    train_data = data_prep()
    a = train_data.shape[1]
    y = train_data[:, a-1:a]
    X = train_data[:, 0:a-1]/100
    m = X.shape[0]
    n = X.shape[1]
    params = initialize_params(n)
    h_params = initialize_hyper_params()
    weights = graident_descent(X,y,params,h_params)
    pickle_out = open("weights.pickle","wb")
    #Saving the weights obtained after training using pickle
    saved_weights = pickle.dump(weights,pickle_out)
    pickle_out.close()
