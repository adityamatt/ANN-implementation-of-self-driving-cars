from src import *
from Utility import *
import pickle
import sys

######Compilation:
#python main.py <iteration> <batch_size> <learning_rate>

######
##Create Dataset
#X,Y=get_data_set()
#print "DataSet Created"
#X,Y=randomize(X,Y)

#Save the randomized data_set
#pickle.dump([X,Y], open("randomized_data_set.txt", "wb"))  

#load the randomized data_set


[X,Y]=pickle.load(open("randomized_data_set.txt", "rb"))

print "Randomized Data Set Loaded"

#SPLIT IS 80-20

training_size=(80*X.shape[0]/100) #17599
validation_size=(20*X.shape[0]/100) #4399


####Defining Variables
iteration=1000
mini_batch_size=32
drop_out=0.0

learning_rate=0.01

iteration=int(sys.argv[1])
mini_batch_size=int(sys.argv[2])
learning_rate=float(sys.argv[3])

if len(sys.argv)>4:
    drop_out=float(sys.argv[4])


num_batches=(training_size/mini_batch_size)-1
#Ending is at mini_batch_size*num_batches
Training_X=X[0:num_batches*mini_batch_size]
Training_Y=Y[0:num_batches*mini_batch_size]

Validation_X=X[num_batches*mini_batch_size+1:]
Validation_Y=Y[num_batches*mini_batch_size+1:]
    
mean=np.mean(Training_X,axis=0)
std=np.std(Training_X,axis=0)

Training_X=(Training_X-mean)/std;
Validation_X=(Validation_X-mean)/std;
##################CREATING MODEL
network=model(1024)
network.add_layer(512,"sigmoid",drop_out)
network.add_layer(64 ,"sigmoid",drop_out)
network.add_layer(1  ,"None",drop_out)

####
print "Program Inputs"
print "Epoch:",iteration,"\tBatch Size:",mini_batch_size,"\tlearning_rate:",learning_rate
print "Number of Batches:",num_batches
print "Size of Training:",Training_X.shape[0]
print "Size of Validation:",Validation_X.shape[0]
print "Drop out value: ",drop_out
####

training_error_list=list()
test_error_list=list()
epoch=list()
for iter_value in range(iteration):
    for i in range(num_batches):
        batch=Training_X[mini_batch_size*i:mini_batch_size*i+mini_batch_size]
        batch_expected_output=Training_Y[mini_batch_size*i:mini_batch_size*i+mini_batch_size]
        batch_output=network.forward_pass(batch)
        error=batch_output-batch_expected_output
        error=error/mini_batch_size
        network.backward_pass(learning_rate,error)
    
    validation_size=Validation_X.shape[0]
    training_size=Training_X.shape[0]
    
    Validation_Error=network.forward_pass(Validation_X)-Validation_Y
    Validation_Error=np.matmul(Validation_Error.T,Validation_Error)
    Validation_Error=Validation_Error/validation_size
    
    Training_Error=network.forward_pass(Training_X)-Training_Y
    Training_Error=np.matmul(Training_Error.T,Training_Error)
    Training_Error=Training_Error/training_size
    
    training_error_list.append(Training_Error[0][0])
    test_error_list.append(Validation_Error[0][0])
    epoch.append(iter_value+1)
    network.clear_stored_values()
    if ((iter_value)%10==0):
        print "Epoch:",iter_value+1,"Validation Error:",Validation_Error[0][0],"\tTraining Error:",Training_Error[0][0]
    
training_file_name="Training_Epoch"+str(iteration)+"_batch_size"+str(mini_batch_size)+"_alpha"+str(learning_rate)
test_file_name="Test_Epoch"+str(iteration)+"_batch_size"+str(mini_batch_size)+"_alpha"+str(learning_rate)
if (drop_out>0):
    training_file_name=training_file_name+"_dropout"+str(drop_out)
    test_file_name=test_file_name+"_dropout"+str(drop_out)
pickle.dump(training_error_list, open(training_file_name, "wb"))
pickle.dump(test_error_list, open(test_file_name, "wb"))
#    network.backward_pass(batch_error,learning_rate)

#print network.layer[0].weight
#print network.layer[1].weight
#print network.layer[2].weight



##print error
#error=np.asarray([error])

#print error
#print error.shape
#print network.layer[0].forwarded.shape
#print network.layer[1].forwarded.shape
#print network.layer[2].forwarded.shape
#network.backward_pass(error,0.001)
