#import cv2
import os
import numpy as np
import random
from scipy.special import expit as sigmoid

def get_data_set():
    directory="./Labs/steering"
    data_file="./Labs/steering/data.txt"
    image_output_dict=dict()
    with open (data_file) as dic:
        content=dic.readlines()
    content = [x.strip() for x in content]
    for c in content:
        c=str(c)
        c=c.split("\t")
        image_output_dict[str(c[0][2:])]=c[1]
    
    num_jpg=0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            num_jpg=num_jpg+1
    k=0
    X=np.zeros((num_jpg,1024))
    Y=np.zeros((num_jpg,1))
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image=cv2.imread(str("./Labs/steering/"+filename),cv2.IMREAD_GRAYSCALE)
            image = np.float32(image)
            tmp=image.flatten()
            X[k]=tmp
            Y[k]=[float(image_output_dict[filename])]
            k=k+1
            print k
    X=np.array(X)
    Y=np.array(Y)
    return X,Y


#A function to randomize the order
def randomize(X1,Y1):
    tmp=np.column_stack((X1,Y1))
    Y1=tmp.T[X1.shape[1]]
    X1=tmp.T[0:X1.shape[1]]
    Y1=Y1.reshape(len(Y1),1)
    X1=X1.T
    return X1,Y1
#Returns a matrix of MXN with equally spaced values between left to right
def getRandomMatrix(m,n,left,right):
    output=np.linspace(left,right,(m-1)*n,dtype=float)
    tmp=np.zeros((1,n),dtype=float)
    output = np.concatenate((tmp,output),axis=None)
    output=output.reshape(m,n)
    return output
    
#Choosing activation
def activation(x,a_type):
    if (a_type=="sigmoid"):
        return sigmoid(x)
    if (a_type=="None"):
        return x

#Choosing activation derivative
def activation_derivative(x,a_type):
    if (a_type=="sigmoid"):
        return sigmaprime(x)
    if (a_type=="None"):
        return np.ones(x.shape)

def sigmaprime(x):
    return x*(1-x)

#Utility function to create a drop out 1-0 layer of size+1 ,1 is for bias at start
def drop_out_layer(fract,size):
    if (fract==0):
        a=np.ones(size+1)
        a=a.reshape(len(a),1)
        return a.T
        
    count_zero=int(round(size*fract))
    count_one=size-count_zero
    row_one=np.ones(count_one)
    row_zero=np.zeros(count_zero)
    
    const=int(round(1/(1-fract)))
    row_one=const*row_one
    
    tmp= np.random.permutation(np.hstack((row_one,row_zero)))
    tmp=np.insert(tmp,0,1)
    tmp=tmp.reshape(len(tmp),1)
    return tmp.T

