import numpy as np
import random
from layer import *
from copy import deepcopy

class model:
    #layers             to count the number of layers
    #layer              to store the layers
    #input_size         to store the size of input layer
    #last_layer_size    to store the last layer size

    
    def __init__(self,input_size):
        self.layers=0
        self.layer=list()
        self.input_size=input_size
        self.last_layer_size=self.input_size
        
    #last_layer_size is input
    def add_layer(self,layer_size,a_type,drp=0.0):
        self.layer.append(layer(self.last_layer_size,layer_size,a_type,drp))
        self.layers=self.layers+1
        self.last_layer_size=layer_size
    
    def forward_pass(self,input_vector):
        tmp=deepcopy(input_vector)
        k=tmp.shape[0]
        bias_identity=np.ones((k,1))
        tmp=np.column_stack((bias_identity,tmp))
        for i in range(self.layers):
            tmp=self.layer[i].f_pass(tmp)
            if i!=self.layers-1:
                k=tmp.shape[0]
                bias_identity=np.ones((k,1))
                tmp=np.column_stack((bias_identity,tmp))
        return tmp
    
    def clear_stored_values(self):
        for i in range(self.layers):
            self.layer[i].forwarded=list()
            self.layer[i].forwarded=np.asarray(self.layer[i].forwarded)
            self.layer[i].output=list()
            self.layer[i].output=np.asarray(self.layer[i].output)
            
            
    def backward_pass(self,learning_rate,error):
        W1=self.layer[0].weight[1:,:]
        W2=self.layer[1].weight[1:,:]
        W3=self.layer[2].weight[1:,:]
        
        I1=self.layer[0].forwarded
        I2=self.layer[1].forwarded
        I3=self.layer[2].forwarded
        
        O1=self.layer[0].output
        O2=self.layer[1].output
        O3=self.layer[2].output
        
        P1=(activation_derivative(O1,self.layer[0].activation_type)).T
        P2=(activation_derivative(O2,self.layer[1].activation_type)).T
        P3=(activation_derivative(O3,self.layer[2].activation_type)).T
        
        D3=error.T
        D2=np.multiply(np.matmul(W3,D3),P2)
        D1=np.multiply(np.matmul(W2,D2),P1)
         
        d_W3=(np.matmul(D3,I3)).T
        d_W2=(np.matmul(D2,I2)).T
        d_W1=(np.matmul(D1,I1)).T
        
#        print O3.shape,"	",P3.shape,"	",D3.shape,"\t",I3.shape,"	",W3.shape,"\t",d_W3.shape
#        print O2.shape,"	",P2.shape,"	",D2.shape,"\t",I2.shape,"	",W2.shape,"\t",d_W2.shape
#        print O1.shape,"	",P1.shape,"	",D1.shape,"\t",I1.shape,"	",W1.shape,"\t",d_W1.shape
        
        self.layer[2].update_weight(learning_rate,d_W3)
        self.layer[1].update_weight(learning_rate,d_W2)
        self.layer[0].update_weight(learning_rate,d_W1)
        self.clear_stored_values()
        
