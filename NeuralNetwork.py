import numpy as np
from matplotlib import pyplot
import scipy.special,scipy.optimize,scipy.io,scipy.misc
import PIL.Image
import json

def displayData(X, theta = None):
    rows,cols = 10,10
    block = 20
    rand_indices = np.random.permutation(5000)[0:rows*cols]
    display = np.zeros((block*rows, block*cols))
    counter = 0
    for i in range(rows):
        for j in range(cols):
            x=j*block
            y=i*block
            display[x:x+block, y:y+block]=X[rand_indices[counter]].reshape(block,block).T
            counter+=1
    img = scipy.misc.toimage(display)
    figure = pyplot.figure()
    axes = figure.add_subplot(1,1,1)
    axes.imshow(img)
    if theta is not None:
        m = X.shape[0]
        result_matrix = []
        X = np.c_[np.ones((m,1)),X]
        for i in rand_indices:
            result = np.argmax(theta.T.dot(X[i]))+1
            result_matrix.append(result)
        result_matrix = np.array(result_matrix).reshape(rows,cols).T
        print result_matrix
            
    pyplot.show()

def sigmoid(p):
    return scipy.special.expit(p)

def sigmoidGradient(p):
    return sigmoid(p)*(1-sigmoid(p))

def recodeLabel(y,k):
    m=y.shape[0]
    out = np.zeros((k,m))
    for i in range(m):
        out[y[i]-1,i] = 1
    return out

def randInitializeWeights(rows,cols):
    e=0.12
    theta = np.random.random((cols,rows+1))*2*e-e
    return theta

def paramUnroll(params,input_layer,hidden_layer,labels):
    theta1_elems = (input_layer+1)*hidden_layer
    theta1_size = (input_layer+1,hidden_layer)
    theta2_size = (hidden_layer+1,labels)
    theta1 = params[:theta1_elems].T.reshape(theta1_size).T
    theta2 = params[theta1_elems:].T.reshape(theta2_size).T
    return theta1, theta2

def feedForward(theta1,theta2,X,X_bias = None):
    ones_row = np.ones((1,X.shape[0]))
    a1 = np.r_[ones_row, X.T] if X_bias is None else X_bias
    z2 = theta1.dot(a1)
    a2 = sigmoid(z2)
    a2 = np.r_[ones_row, a2]
    z3 = theta2.dot(a2)
    a3 = sigmoid(z3)
    return a1,a2,a3,z2,z3   
    
def computeCost(params,input_layer,hidden_layer,labels,X,y,lamda,yk=None,X_bias=None):
    m,n = np.shape(X)
    theta1, theta2 = paramUnroll(params,input_layer,hidden_layer,labels)
    a1,a2,a3,z2,z3 = feedForward(theta1,theta2,X,X_bias)
    if yk is None:
        yk = recodeLabel(y, labels)
    term1 = np.log(a3)*(-yk)
    term2 = np.log(1-a3)*(1-yk)
    first = (np.sum(term1 - term2)/m)
    second = (np.sum(theta1[:,1:]**2)+np.sum(theta2[:,1:]**2) )* lamda/(2*m)
    return first + second

def computeGradient(params,input_layer,hidden_layer,labels,X,y,lamda,yk = None,X_bias = None):
    m,n = np.shape(X)
    theta1, theta2 = paramUnroll(params,input_layer,hidden_layer,labels)
    a1,a2,a3,z2,z3 = feedForward(theta1,theta2,X,X_bias)

    if yk is None:
        yk = recodeLabel(y, labels)
    
    error3 = a3 - yk
    error2 = theta2.T.dot(error3) * sigmoidGradient(np.r_[np.ones((1,m)),z2])
    error2 = error2[1:,:]
    D1 = error2.dot(a1.T)/m
    D2 = error3.dot(a2.T)/m
    D1[:,1:] = D1[:,1:] + (theta1[:,1:] * lamda / m)
    D2[:,1:] = D2[:,1:] + (theta2[:,1:] * lamda / m)
    #reshape array D1,D2 to a list
    D = np.array([D1.T.reshape(-1).tolist()+D2.T.reshape(-1).tolist()]).T
    return np.ndarray.flatten(D)

def nnCostFunction(params,input_layer,hidden_layer,labels,X,y,lamda,yk = None,X_bias = None):
    m,n = np.shape(X)
    theta1, theta2 = paramUnroll(params,input_layer,hidden_layer,labels)
    a1,a2,a3,z2,z3 = feedForward(theta1,theta2,X)

    #if yk is None:
    yk = recodeLabel(y, labels)
    
    term1 = np.log(a3)*(-yk)
    term2 = np.log(1-a3)*(1-yk)
    first = (np.sum(term1 - term2)/m)
    second = (np.sum(theta1[:,1:]**2)+np.sum(theta2[:,1:]**2) )* lamda/(2*m)
    cost = first + second

    error3 = a3 - yk
    error2 = theta2.T.dot(error3) * sigmoidGradient(np.r_[np.ones((1,m)),z2])
    error2 = error2[1:,:]
    D1 = error2.dot(a1.T)/m
    D2 = error3.dot(a2.T)/m
    D1[:,1:] = D1[:,1:] + (theta1[:,1:] * lamda / m)
    D2[:,1:] = D2[:,1:] + (theta2[:,1:] * lamda / m)
    #reshape array D1,D2 to a list
    D = np.array([D1.T.reshape(-1).tolist() + D2.T.reshape(-1).tolist()]).T
    return cost,D

def gradChecking(theta,input_layer,hidden_layer,labels,X,y,lamda):
    e = 1e-4
    numGrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    num_elements = theta.shape[0]
    yk = recodeLabel(y,labels)
    for p in range(num_elements):
        perturb[p] = e
        left = computeCost(theta+perturb, input_layer,hidden_layer,labels,X,y,lamda,yk)
        right = computeCost(theta-perturb, input_layer,hidden_layer,labels,X,y,lamda,yk)
        numGrad[p] = left-right/(2*e)
        perturb[p]=0
    return numGrad

def predict(X,theta1,theta2):
    a1 = np.r_[np.ones((1,1)), X.reshape(X.shape[0],1)]
    z2 = theta1.dot(a1)
    a2 = sigmoid(z2)
    a2 = np.r_[np.ones((1,1)), a2]
    z3 = theta2.dot(a2)
    a3 = sigmoid(z3)
    #the indice corresponding to maximum argument will be the prediction
    return np.argmax(z3) + 1
    

def part7():
    images = scipy.io.loadmat("ex4data1.mat")
    X,y = images['X'],images['y']
    m,n = np.shape(X)
    
    weights = scipy.io.loadmat("ex4weights.mat")
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    params = np.r_[theta1.T.flatten(),theta2.T.flatten()]
    lamda = 1
    input_layer = 400
    output_layer = 10
    hidden_layer = 25
    numGrad = gradChecking(params,input_layer,hidden_layer,output_layer,X,y,lamda)
    print numGrad

def nNetwork(trainingData,filename):

    lamda = 1
    input_layer = 1200
    output_layer = 10
    hidden_layer = 25
    X=trainingData[0]
    y=trainingData[1]
    theta1 = randInitializeWeights(1200,25)
    theta2 = randInitializeWeights(25,10)
    m,n = np.shape(X)
    yk = recodeLabel(y,output_layer)
    theta = np.r_[theta1.T.flatten(), theta2.T.flatten()]

    X_bias = np.r_[np.ones((1,X.shape[0])), X.T]
    #conjugate gradient algo
    result = scipy.optimize.fmin_cg(computeCost,fprime=computeGradient,x0=theta,args=(input_layer,hidden_layer,output_layer,X,y,lamda,yk,X_bias),maxiter=100,disp=True,full_output=True )
    print result[1]  #min value
    theta1,theta2 = paramUnroll(result[0],input_layer,hidden_layer,output_layer)
    counter = 0
    for i in range(m):
        prediction = predict(X[i],theta1,theta2)
        actual = y[i]
        if(prediction == actual):
            counter+=1
    print  str(counter *100/m) + '% accuracy'

    scipy.io.savemat(filename, mdict ={'theta1':theta1,'theta2':theta2})
    

