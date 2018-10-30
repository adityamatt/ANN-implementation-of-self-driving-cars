import matplotlib.pyplot as plt
import pickle
import sys

folder_link="./stored_experiments/"

file_15="Test_Epoch5000_batch_size64_alpha0.01"                         #Exp 1,2
file_16="Training_Epoch5000_batch_size64_alpha0.01"                     #Exp 1,2
file_1="Test_Epoch1000_batch_size128_alpha0.01"                         #Exp 2
file_2="Test_Epoch1000_batch_size32_alpha0.01"                          #Exp 2
file_8="Training_Epoch1000_batch_size32_alpha0.01"                      #Exp 2
file_7="Training_Epoch1000_batch_size128_alpha0.01"                     #Exp 2

file_4="Test_Epoch1000_batch_size64_alpha0.001_dropout0.5"              #Exp 3
file_10="Training_Epoch1000_batch_size64_alpha0.001_dropout0.5"         #Exp 3

file_3="Test_Epoch1000_batch_size64_alpha0.001"                         #Exp 4
file_9="Training_Epoch1000_batch_size64_alpha0.001"                     #Exp 4
file_5="Test_Epoch1000_batch_size64_alpha0.005"                         #Exp 4
file_6="Test_Epoch1000_batch_size64_alpha0.05"                          #Exp 4
file_11="Training_Epoch1000_batch_size64_alpha0.005"                    #Exp 4
file_12="Training_Epoch1000_batch_size64_alpha0.05"                     #Exp 4
file_13="Test_Epoch1000_batch_size64_alpha0.01_dropout0.5"              #Exp 4
file_14="Training_Epoch1000_batch_size64_alpha0.01_dropout0.5"          #Exp 4

file_1=folder_link+file_1
file_2=folder_link+file_2
file_3=folder_link+file_3
file_4=folder_link+file_4
file_5=folder_link+file_5
file_6=folder_link+file_6
file_7=folder_link+file_7
file_8=folder_link+file_8
file_9=folder_link+file_9
file_10=folder_link+file_10
file_11=folder_link+file_11
file_12=folder_link+file_12
file_13=folder_link+file_13
file_14=folder_link+file_14
file_15=folder_link+file_15
file_16=folder_link+file_16

#VALIDATIOn-RED TRAINING_GREEN
experiment_num=int(sys.argv[1])

#A plot of sum of squares error on the training and validation set as a function of
#training iterations (for 5000 epochs) with a learning rate of 0.01. (no dropout,
#minibatch size of 64).
if (experiment_num==1):
    epoch=range(1,5001)
    Training_Error=pickle.load(open(file_16,"rb"))
    Validation_Error=pickle.load(open(file_15,"rb"))
    plt.plot(epoch,Training_Error,label="Training Error",color='green')
    plt.plot(epoch,Validation_Error,label="Validation Error",color='red')
    plt.xlabel("Number of epochs")
    plt.ylabel("Mean Square Error")
    plt.title("Training,Validation Error vs number of epochs with no dropout,batch size = 64")
    plt.legend()
    plt.show()
    plt.clf()

elif (experiment_num==2):
    epoch=range(1,1001)
    Training_error_32=pickle.load(open(file_8,"rb"))
    Validation_error_32=pickle.load(open(file_2,"rb"))
    
    Training_Error_64=pickle.load(open(file_16,"rb"))
    Validation_Error_64=pickle.load(open(file_15,"rb"))
    Training_Error_64=Training_Error_64[0:1000]
    Validation_Error_64=Validation_Error_64[0:1000]
    
    
    Training_error_128=pickle.load(open(file_7,"rb"))
    Validation_error_128=pickle.load(open(file_1,"rb"))
    
    plt.plot(epoch,Training_error_32,label="Training Error",color='green')
    plt.plot(epoch,Validation_error_32,label="Validation Error",color='red')
    plt.xlabel("Number of epochs")
    plt.ylabel("Mean Square Error")
    plt.title("Training,Validation Error vs number of epochs with no dropout,batch size = 32")
    plt.legend()
    plt.show()
    plt.clf()
    
    plt.plot(epoch,Training_Error_64,label="Training Error",color='green')
    plt.plot(epoch,Validation_Error_64,label="Validation Error",color='red')
    plt.xlabel("Number of epochs")
    plt.ylabel("Mean Square Error")
    plt.title("Training,Validation Error vs number of epochs with no dropout,batch size = 64")
    plt.legend()
    plt.show()
    plt.clf()
    
    plt.plot(epoch,Training_error_128,label="Training Error",color='green')
    plt.plot(epoch,Validation_error_128,label="Validation Error",color='red')
    plt.xlabel("Number of epochs")
    plt.ylabel("Mean Square Error")
    plt.title("Training,Validation Error vs number of epochs with no dropout,batch size = 128")
    plt.legend()
    plt.show()
    plt.clf()
    #A plot of sum of squares error on the training and validation set as a function of
    #training iterations (for 1000 epochs) with a fixed learning rate of 0.01 for three
    #minibatch sizes 32,64,128
    
elif (experiment_num==3):
    epoch=range(1,1001)
    Training_error=pickle.load(open(file_10,"rb"))
    Validation_error=pickle.load(open(file_4,"rb"))
    plt.plot(epoch,Training_error,label="Training Error",color='green')
    plt.plot(epoch,Validation_error,label="Validation Error",color='red')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Square Error")
    plt.title("Training,Validation Error vs Number of epochs with dropout probability= 0.5,learning rate=0.001")
    plt.legend()
    plt.show()
    plt.clf()
    
    Training_error=pickle.load(open(file_14,"rb"))
    Validation_error=pickle.load(open(file_13,"rb"))
    plt.plot(epoch,Training_error,label="Training Error",color='green')
    plt.plot(epoch,Validation_error,label="Validation Error",color='red')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Square Error")
    plt.title("Training,Validation Error vs Number of epochs with dropout probability= 0.5,learning rate=0.001")
    plt.legend()
    plt.show()
    
    #A plot of sum of squares error on the training and validation set as a function of
    #training iterations (for 1000 epochs) with a learning rate of 0.001, and dropout
    #probability of 0.5 for the first, second and third layers
    
    

elif (experiment_num==4):
    #A plot of sum of squares error on the training and validation set as a function of
    #training iterations (for 1000 epochs) with different learning rates  0.05, 0.001,
    #0.005 no drop out, minibatch size 64
    
    #alpha1 0.05
    #alpha2 0.001
    #alpha3 0.005
    
    epoch=range(1,1001)
    Training_error_alpha1=pickle.load(open(file_12,"rb"))    
    Training_error_alpha2=pickle.load(open(file_9,"rb"))
    Training_error_alpha3=pickle.load(open(file_11,"rb"))
    
    Test_error_alpha1=pickle.load(open(file_6,"rb"))
    Test_error_alpha2=pickle.load(open(file_3,"rb"))
    Test_error_alpha3=pickle.load(open(file_5,"rb"))
    
    plt.plot(epoch,Training_error_alpha1,label="Training Error",color='green')
    plt.plot(epoch,Test_error_alpha1,label="Validation_Error",color='red')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Square Error")
    plt.title("Training,Validation Error vs Number of epochs with No drop out,learning_rate=0.05")
    plt.legend()
    plt.show()
    plt.clf()
    
    plt.plot(epoch,Training_error_alpha2,label="Training Error",color='green')
    plt.plot(epoch,Test_error_alpha2,label="Validation_Error",color='red')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Square Error")
    plt.title("Training,Validation Error vs Number of epochs with No drop out,learning_rate=0.001")
    plt.legend()
    plt.show()
    plt.clf()
    
    plt.plot(epoch,Training_error_alpha3,label="Training Error",color='green')
    plt.plot(epoch,Test_error_alpha3,label="Validation_Error",color='red')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Square Error")
    plt.title("Training,Validation Error vs Number of epochs with No drop out,learning_rate=0.005")
    plt.legend()
    plt.show()
    plt.clf()
else:
    print "Please enter a valid Experiment number"
