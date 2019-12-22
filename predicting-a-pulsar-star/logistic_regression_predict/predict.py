import pickle
import numpy as np
#Getting the weights after training using pickle
pickle_in = open("weights.pickle","rb")
weights = pickle.load(pickle_in)

def data_prep():
 #preparing the test data (i.e the bottom 20 percent of the whole data)
 data = np.genfromtxt('pulsar_stars.csv', delimiter = ',', skip_header=1)
 m = data.shape[0]
 n = int(m  * 0.2)
 test_data = data[m-n:m]
 return test_data



def sigmoid(z):
 return 1/(1+np.exp(-1 * z))



def predict(test_data, weights):
    #Getting our predictions on the test data and printing the accuracy

    W = weights["W"]
    b = weights["b"]
    a = test_data.shape[1]
    X = test_data[:,0:a-1]/100
    y = test_data[:,a-1:a]
    p_v = (sigmoid(np.dot(X,W) + b) > 0.5)
    print("---Accuracy---")
    print(np.sum(p_v)/np.sum(y) * 100)



if __name__ == "__main__":
    test_data = data_prep()
    predict(test_data,weights)