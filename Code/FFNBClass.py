# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 21:09:35 2020

@author: chandra Naveen Kumar
"""

""" Importing the required libraries """
import math
from random import randint

""" Importing the required libraries """


""" Dot product calculation fucntion for matrix multiplication of list of lists of weights & inputs  """
def dot(v1,v2):                                     
    i = 0
    Out  = []
    for i in range(len(v2)):
        temp = v2[i]
        Out.append(sum(x*y for x,y in zip(v1,temp)))
    return Out

""" Function used for randomly normalizing the weights between o and 1"""
def rand(x,y):                                      
    i = 0
    Intialized_array = []
    for i in range(x):
        j = 0
        temp = []
        for j in range(y):
            temp.append(float(randint(1,10)/10))
            j += 1
        Intialized_array.append(temp)
        i += 1
    return Intialized_array
""" Neural network class for intializing a new network object all the actions are defined as methods"""

class Neuron_Network() :
    
    def __init__(self,N_Inputs,N_hiddenNeurons,N_Outputs):
        """ Initializing the network and creating weights matrices with the required demisions specified by user 
            also intializing various parameters required for Forward and Backward propagation while training the model"""
        self.Input_Neuron_nodes = N_Inputs 
        self.Hidden_Neuron_Nodes = N_hiddenNeurons
        self.Output_Neurons_Nodes = N_Outputs
        self.Hidden_Layer_bias = [1]                                                                                           # Intializing the bias node value for Input layer
        self.learning_rate = 0.9
        self.momentum_rate = 0.1
        self.Outlayer_Prevwu = [[0 for col in range(self.Hidden_Neuron_Nodes+1)] for row in range(self.Output_Neurons_Nodes)]  # weight vector containing the previous step output layer delta weights 
        self.HidLayer_Prevwu = [[0 for col in range(self.Input_Neuron_nodes+1)] for row in range(self.Hidden_Neuron_Nodes)]    # weight vector containing the previous step hidden layer delta weights
        self.Hidden_Layer_Weights = rand(self.Hidden_Neuron_Nodes,self.Input_Neuron_nodes+1)                                   # The baise weight is also selected randomly and part of the matrix generated
        self.Output_Layer_bias = [1]                                                                                           # Intializing the bias node value for Hidden layer
        self.Output_Layer_Weights = rand(self.Output_Neurons_Nodes, self.Hidden_Neuron_Nodes+1)                                # The baise weight is also selected randomly and part of the matrix generated
        
    def Activation_func(self,x):
        """sigmoid activation function to be used in both Hidden and Output layers """
        self.Lamda_value = 0.9
        i = 0
        self.activ = [] 
        for i in range(len(x)):
            self.activ.append(1/(1+math.exp(-self.Lamda_value*x[i])))
            i = i + 1
        return self.activ
    
    def Deriv_Activ_func(self,x):
        """ First order derivative of sigmoid activation function """ 
        self.Deriv_value = []
        i = 0
        for i in range(len(x)):
            self.Deriv_value.append((self.Lamda_value*x[i])*(1-x[i]))
            i = i+1
        return self.Deriv_value
    def Local_gradient_out(self,x,y):
        """ A function for calculating the local gradients of the output layer while back propagation"""
        self.Local_grad_out = []
        i = 0
        for i in range(len(x)):
            self.Local_grad_out.append(x[i]*y[i])
            i = i+1
        return self.Local_grad_out
    def Local_gradient_hid(self,Dervative_values,Outlayer_LOG,Weights_hiddenLayer):
        """ A function for calculating the local gradients of the Hidden layer while back propagation"""
        local_grad_hid = []
        i = 0
        for i in range(len(Weights_hiddenLayer[0])):
            j = 0
            for j in range(len(Outlayer_LOG)):
                out = Outlayer_LOG[j]*Weights_hiddenLayer[j][i]
                j =j+1
            local_grad_hid.append(Dervative_values[i]*out)
            i = i+1
        return local_grad_hid
    def delta_weight_out(self, Gradient_values,Hid_layer_outs):
        """ Calculating the update value to each weight based on the previously calculated lacal gradient value of the output layer"""
        i = 0
        delta_weights = []
        for i in range(len(Gradient_values)):
            j = 0
            temp =[]
            for j in range(len(Hid_layer_outs)):
                temp.append((self.learning_rate*Gradient_values[i]*Hid_layer_outs[j])+(self.momentum_rate*self.Outlayer_Prevwu[i][j]))
                j = j+1
            delta_weights.append(temp)
            i = i+1
        self.Outlayer_Prevwu = delta_weights
        return delta_weights
    def delta_weight_hid(self, Gradient_values,Inputs):
        """ Calculating the update value to each weight based on the previously calculated lacal gradient value of the Hidden layer"""
        i = 0
        delta_weights = []
        for i in range(len(Gradient_values)):
            j = 0
            temp =[]
            for j in range(len(Inputs)):
                temp.append((self.learning_rate*Gradient_values[i]*Inputs[j])+(self.momentum_rate*self.HidLayer_Prevwu[i][j]))
                j = j+1
            delta_weights.append(temp)
            i = i+1
        self.HidLayer_Prevwu = delta_weights
        return delta_weights
    def Update_weights_Out(self, Current_weights, Correction_values):
        """ Making the necessary corrections to the present weights based on the delta weights calculated for Output Layer"""
        i = 0
        new_weights= []
        for i in range(len(Current_weights)):
            j = 0 
            temp = []
            temp1 = Current_weights[i]
            temp2 = Correction_values[i]
            for j in range(len(temp1)):
                temp.append(temp1[j] + temp2[j])
                j = j+1
            new_weights.append(temp)
            i = i+1
        return new_weights
    
    def Update_weights_Hid(self, Current_weights, Correction_values):
        """ Making the necessary corrections to the present weights based on the delta weights calculated for Hidden Layer Layer"""
        i = 0
        new_weights= []
        for i in range(len(Current_weights)):
            j = 0 
            temp = []
            temp1 = Current_weights[i]
            temp2 = Correction_values[i]
            for j in range(len(temp1)):
                temp.append(temp1[j] + temp2[j])
                j = j+1
            new_weights.append(temp)
            i = i+1
        return new_weights
    def newWeights(self, weightvector):
        """ removing the weight of the bias node while passing weight vector to gradient calculation"""
        new_weights = []
        i = 0
        for i in range(len(weightvector)):
            temp = [0] * (len(weightvector[i])-1)
            j = 0
            for j in range(len(weightvector[i])):
                if j != 0:
                    temp[j-1] = weightvector[i][j]
                j = j+1
            new_weights.append(temp)
            i= i+1
        return new_weights
                
        
    def Train(self, Training_inputs, Training_labels, validation_inputs, validation_labels, Epoches_to_train):
        """ A function to train the network with the passed inputs 
            It takes Training data along with expected outcomes, validation data along with expected outcomes and the user can specifiy maximum epoched to train""" 
        epoch_length = 0
        epoch_test_error  = 0
        Val_error = []
        Train_error = []
        epoch_val_error = 0
        val_inc_count = 0
        prev_epoch_v_error = 0
        epoch_Outlayer_weights = []
        epoch_Hidlayer_weights = []
        """ Intializing all error varaibles to zero and null weight vectors to hold weight values for each epoch """
        while epoch_length < Epoches_to_train:
            """ The training loop is goanna run until end of epoches specified user or unitl early stopping criteria conditions meet which ever happens first """
            train_row = 0
            val_row = 0
            for train_row in range(len(Training_inputs)):
                Training_examples = Training_inputs[train_row]
                Expected_Output   = Training_labels[train_row]
                inputs = self.Hidden_Layer_bias+ Training_examples
                """ Adding the bias value one to the inputs row for feed forward calculation"""
                self.Hidden_Layer_Out = self.Activation_func(dot(inputs,self.Hidden_Layer_Weights))
                Temp_Hidden_outs = self.Output_Layer_bias + self.Hidden_Layer_Out
                """ Adding the bias value one to the calculated hidden layer out values for feed forward calculation"""
                self.Output_Layer_Out = self.Activation_func(dot(Temp_Hidden_outs, self.Output_Layer_Weights))
                self.Observed_error = self.error(Expected_Output)
                """ Calculating error using the expected outcome from data """
                self.Outlayer_SigDerv = self.Deriv_Activ_func(self.Output_Layer_Out)
                self.Hiddenlayer_sigDerv = self.Deriv_Activ_func(self.Hidden_Layer_Out)
                self.OutLayer_LOG = self.Local_gradient_out(self.Outlayer_SigDerv, self.Observed_error)
                """ Calculating the Local Gradient of the output nodes """
                new_weights = self.newWeights(self.Output_Layer_Weights)
                self.Hidden_Layer_LOG =self.Local_gradient_hid(self.Hiddenlayer_sigDerv, self.OutLayer_LOG,new_weights)
                """ Calculating the Local Gradient of the Hidden layer nodes """
                self.OutLayer_UpdateWeights  = self.delta_weight_out(self.OutLayer_LOG, Temp_Hidden_outs)
                self.HidLayer_UpdateWeights = self.delta_weight_hid(self.Hidden_Layer_LOG, inputs)
                self.Corrected_Hid_weights = self.Update_weights_Hid(self.Hidden_Layer_Weights, self.HidLayer_UpdateWeights)
                self.Hidden_Layer_Weights = self.Corrected_Hid_weights
                self.Corrected_Out_weights = self.Update_weights_Out(self.Output_Layer_Weights, self.OutLayer_UpdateWeights)
                self.Output_Layer_Weights = self.Corrected_Out_weights
                """ calculating the weight corrections and updating the weights over all layers"""
                current_error = ((self.Observed_error[0])**2 + (self.Observed_error[1])**2)*0.5
                epoch_test_error = epoch_test_error + current_error                                                          # Error gets accumiliated during each epoch run for train error calculation
                train_row = train_row + 1
            for val_row in range(len(validation_inputs)):
                """ During the validation run we feed forward the input and make prediction to calculate the error but the observed error is not back propagated for weight correction"""
                Training_examples = validation_inputs[val_row]
                Expected_Output   = validation_labels[val_row]
                inputs = self.Hidden_Layer_bias+ Training_examples
                self.Hidden_Layer_Out = self.Activation_func(dot(inputs,self.Hidden_Layer_Weights))
                Temp_Hidden_outs = self.Output_Layer_bias + self.Hidden_Layer_Out
                self.Output_Layer_Out = self.Activation_func(dot(Temp_Hidden_outs, self.Output_Layer_Weights))
                self.Observed_error = self.error(Expected_Output)
                current_error = ((self.Observed_error[0])**2 + (self.Observed_error[1])**2)*0.5
                epoch_val_error = epoch_val_error + current_error
                val_row = val_row + 1
            
            print("The " + str(epoch_length)+ " epoch train error is "+ str(math.sqrt(epoch_test_error/(len(Training_inputs)))))
            print("The " + str(epoch_length)+ " epoch validation error is "+ str(math.sqrt(epoch_val_error/(len(validation_inputs)))))
            epoch_Outlayer_weights.append(self.Output_Layer_Weights)
            epoch_Hidlayer_weights.append(self.Hidden_Layer_Weights)
            """ Appending the weight vectors after each epoch for retrevial in case of early stopping criteria kicks in"""
            if epoch_val_error > prev_epoch_v_error:
                val_inc_count = val_inc_count +1
                """ A counter value for observing incrimental trend in the validation error after each epoch"""
            else:
                val_inc_count = 0
            prev_epoch_v_error = epoch_val_error
            epoch_val_error = math.sqrt(epoch_val_error/len(validation_inputs))
            epoch_test_error = math.sqrt(epoch_test_error/len(Training_inputs))
            Val_error.append(epoch_val_error)
            Train_error.append(epoch_test_error)
            epoch_test_error = 0
            epoch_val_error  = 0
            """ Resetting the current epoch erorr valued to zeros at end of the epoch for next epoch """
            if val_inc_count ==5 :
                """ Training will get stopped if a incremental trend is observed in last five epoches and the weights get replaced with best validation error epochs weights and passed on for testing """
                print("The neural Network Converged at epoch ", epoch_length-4)
                print("The Optimal Hidden Layer Weights",epoch_Hidlayer_weights[epoch_length -5 ])
                print("The Optimal Output Layer Weights", epoch_Outlayer_weights[epoch_length -5])
                self.Output_Layer_Weights = epoch_Outlayer_weights[epoch_length -5]
                self.Hidden_Layer_Weights = epoch_Hidlayer_weights[epoch_length -5 ]
                xvals = range(epoch_length+1)
                break
            epoch_length = epoch_length+1
            
    def Test(self, Test_examples, Test_Output):
        """ During the testing phase all the inputs rows gets feed forwarded and erorr is caluclated """ 
        test_row = 0
        test_error = 0
        test_cur = 0
        for test_row in range(len(Test_examples)):
            inputs = self.Hidden_Layer_bias+ Test_examples[test_row]
            self.Hidden_Layer_Out = self.Activation_func(dot(inputs,self.Hidden_Layer_Weights))
            Temp_Hidden_outs = self.Output_Layer_bias + self.Hidden_Layer_Out
            self.Output_Layer_Out = self.Activation_func(dot(Temp_Hidden_outs, self.Output_Layer_Weights))
            self.Observed_error = self.error(Test_Output[test_row])
            test_cur = ((self.Observed_error[0])**2 + (self.Observed_error[1])**2)*0.5
            test_error = test_error + test_cur
            test_row = test_row +1
        print("The Observed Test Error is ", math.sqrt(test_error/(len(Test_examples))))
    
    def Predict(self, Current_Position):
        """ Function to use just for predicting output velocites for the given X & Y distances from pad"""
        inputs = self.Hidden_Layer_bias+ Current_Position
        self.Hidden_Layer_Out = self.Activation_func(dot(inputs,self.Hidden_Layer_Weights))
        Temp_Hidden_outs = self.Output_Layer_bias + self.Hidden_Layer_Out
        self.Output_Layer_Out = self.Activation_func(dot(Temp_Hidden_outs, self.Output_Layer_Weights))
        return self.Output_Layer_Out
    def error(self, Training_Outputs):
        """ Error calculation function """
        self.error_value = []
        i = 0
        for i in range(len(Training_Outputs)):
            self.error_value.append(Training_Outputs[i] - self.Output_Layer_Out[i])
            i +=1
        return self.error_value 