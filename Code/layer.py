import numpy as np
import random
from Utility import *
from copy import deepcopy

#DEFINE VARIABLES
left_boundary=-0.01
right_boundary=0.01
    
class layer:
    #input_size
    #activation_type
    #weight
    #Forwarded
    #Output
    #dropout
    
    #constructor
    def __init__(self,input_size,output_size,activation,drp=0.0):
        self.input_size=input_size
        self.activation_type=activation
        self.weight=getRandomMatrix(input_size+1,output_size,left_boundary,right_boundary)
        self.forwarded=list()
        self.forwarded=np.asarray(self.forwarded)
        self.output=list()
        self.output=np.asarray(self.output)
        self.dropout=drp
#        self.drp_vec=drop_out_layer(drp,input_size)
        
    def f_pass(self,input_vector):
        input_vector=np.array(input_vector)
        drp_vec=drop_out_layer(self.dropout,self.input_size)
        input_vector=input_vector*drp_vec
        self.forwarded=input_vector
        out=activation(np.matmul(input_vector,self.weight),self.activation_type)
        self.output=out
        return out
    
    #Update the weights
    def update_weight(self,learning_rate,del_w):
        self.weight=self.weight-learning_rate*del_w

        
        
