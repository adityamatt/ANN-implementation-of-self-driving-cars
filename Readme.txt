Name:Aditya Tiwari
Entry:2016csb1029
###########################
Library Used:
matplotlib
pickle
numpy
opencv      (This is optional if need to construct the dataset)

##########################

I have converted the dataset of images (read opencv library) to randomized data set and pickled it which is a 549 mb file
You may download it from:   https://drive.google.com/file/d/15jpYFUBDXY9sOXI_mRkIIpQk6RYtMJmy/view?usp=sharing
and paste it in the folder of code
##########################
To reconstruct the dataset from scratch:
    uncomment the lines:11-13 in main.py
                        1 in Utility.py
###########################
I have stored the outputs of all experiments as pickled file in a folder stored_experiments
To reconstruct the graphs:
    python graph.py <experiment_no.>
e.g
    python graph.py 1
    python graph.py 2
    python graph.py 3
    python graph.py 4
To reconstruct the experiment:
    python main.py <iteration> <batch_size> <learning_rate>(optional)
for experiment 1:
    python main.py 5000 64 0.01

for experiment 2:
    python main.py 1000 32 0.01
    python main.py 1000 64 0.01
    python main.py 1000 128 0.01

for experiment 3:
    python main.py 1000 64 0.001 0.5

for experiment 4:
    python main.py 1000 64 0.05
    python main.py 1000 64 0.005
    python main.py 1000 64 0.001

Note: All these experiments take time
    When done on Lab computers(with 8 cores) each epoch took aroudn 30 seconds

Each experiment when ends creates two pickled file(of small kbs) with name format
"Training_Epoch"+str(iteration)+"_batch_size"+str(mini_batch_size)+"_alpha"+str(learning_rate)
"Test_Epoch"+str(iteration)+"_batch_size"+str(mini_batch_size)+"_alpha"+str(learning_rate)

Paste the files in the stored_experiments folder before running graph.py
########################
Contact:
2016csb1029@iitrpr.ac.in        (Valid till 2020)
aditya.tiwarics@gmail.com
github.com/adityamatt
