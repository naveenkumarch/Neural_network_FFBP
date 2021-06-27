# -*- coding: utf-8 -*-
"""
Created on Mon Nov  30 12:11:24 2020

@author: chandra Naveen Kumar
"""
import math
from FFNBClass import Neuron_Network
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
""" Reading data from the csv file and addig it to  list varaible """
data = []
with open(r'C:\Users\chand\Desktop\Labs\Neural_Networks_Lab\ce889assignment_v2\ce889assignment-master\ce889_dataCollection.csv') as csvfile:
        reader = csv.reader(csvfile, quoting= csv.QUOTE_NONNUMERIC)
        for row in reader:
            data.append(row)
""" Finding the min and max values of every column in the loaded Data and normalize the data between 0 & 1"""
j = 0
while j<len(data[0]):
    i = 0
    max_value = data[0][j]
    min_value = data[0][j]
    """ finding the min and max values of the column"""
    while i < len(data):
        if data[i][j]>max_value:
            max_value = data[i][j]
        elif data[i][j]<min_value:
            min_value = data[i][j]
        else :
            pass
        i = i+1 
    n = 0
    """ Normalizing data on that column using the min and max values"""
    while n < len(data):
        data[n][j] = (data[n][j]-min_value)/(max_value - min_value)
        n = n+1
    print ("Max value of" +str(j) +"th element is "+str(max_value))
    print ("Min value of" +str(j)+ "th element is "+str(min_value))
    j = j +1
input_data = []
input_lable= []
Row = 0
for Row in range(len(data)):
    temp1 = []
    temp2 = []
    temp1.append(data[Row][0])
    temp1.append(data[Row][1])
    temp2.append(data[Row][2])
    temp2.append(data[Row][3])
    input_data.append(temp1)
    input_lable.append(temp2)
    Row = Row +1
""" Partioning data into input vector and labels vector """
        
Data_DF = DataFrame(data,columns=['X_dist','Y_dist','X_vel','Y_vel'])
print(Data_DF.info())
print(Data_DF.isnull().sum())


if __name__ == "__main__":
    """ Initializing the neural network with the required no of input, hidden & output nodes"""
    N_NET = Neuron_Network(2, 12, 2)
    """ Split data into train and test+val sets with 70% for training and 30% for testing & validation"""
    X_train,X_TV,Y_train,Y_TV = train_test_split(input_data,input_lable,test_size = 0.3, shuffle = True)
    """ Spliting the above test & val data into half so each would amount of 15 % of total data"""
    X_test,X_val,Y_test,Y_val = train_test_split(X_TV,Y_TV, test_size = 0.5 ,shuffle = True)
    """ Train the neural network using the train data and validation data"""
    N_NET.Train(X_train, Y_train, X_val, Y_val, 750)
    """ Once the training has been completed and best validation loss was achieved then test the model on the test data set which it hasn't seen before """
    N_NET.Test(X_test,Y_test)
    pass