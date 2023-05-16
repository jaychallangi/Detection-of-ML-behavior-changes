import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.special
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error


def showpicture(path):
    test_data_file = open(path, 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    for index in range(10):
        record = test_data_list[index]
        all_values = record.split(',')
        image_array = numpy.asfarray( all_values[1:]).reshape((28,28))
        # change image to 28 * 28 matrix to show
        plt.imshow( image_array, cmap='Greys',interpolation='None')
        plt.show()
# to look what the pictue look like 
#showpicture("test.csv")


# Below is the 2-layer neural network model
class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #The inodes, hnodes, onodes and lr defined below are only used inside this neuralNetwork class, which are equivalent to internal parameters, and are called directly with self when used
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        '''
        We set the center of the normal distribution to 0.0. The expression for the standard variance associated with the nodes in the next layer, in Python form, is pow(self.hnodes, -0.5). 
        Simply put, this expression is the -0.5 power of the number of nodes. The last argument is the size we want the numpy array to be.
        '''
        self.wih = numpy.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes, self.hnodes))
        # Initialize the weight coefficients for each layer
        self.activation_function = lambda x: scipy.special.expit(x)#Define the activation function
        pass

    def train(self,inputs_list,targets_list):
        # Turn the input data into a column vector
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # Calculate the input of the hidden layer, that is, calculate the product of the weight coefficients between the input layer and the hidden layer and the input data
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calculate the input of the output layer, that is, calculate the product of the weight coefficients between the hidden layer and the output layer and the input data
        final_inputs = numpy.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)
        #Calculation error, this error is the error at the very beginning, that is, the difference between the target value and the data output by the output layer
        output_errors = targets - final_outputs 
        #This is the error back propagation between the output layer to the hidden layer
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # The following is to update the weight parameters between the layers using back propagation of errors
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose (inputs))
        pass

    # Needed to perform our Adversial Attacks
    def lossFunction(self,inputs_list,targets_list):
        # Turn the input data into a column vector
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # Calculate the input of the hidden layer, that is, calculate the product of the weight coefficients between the input layer and the hidden layer and the input data
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calculate the input of the output layer, that is, calculate the product of the weight coefficients between the hidden layer and the output layer and the input data
        final_inputs = numpy.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)
        # Loss Function
        # Calculation error, this error is the error at the very beginning, that is, the difference between the target value and the data output by the output layer

        outputError=mean_absolute_error(final_outputs,targets)
        return outputError
        pass

    def query(self,inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        # Calculate the input of the hidden layer, that is, calculate the product of the weight coefficients between the input layer and the hidden layer and the input data
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Use the activation function to calculate the output of the output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass

    def pretrain(self):
        #The Epoch number is a hyperparameter that defines the number of times the learning algorithm works on the entire training dataset. An Epoch means that each sample in the training dataset has the opportunity to update the internal model parameters
        # If the accuracy is not high enough,  improve echo
        echo = 3
        for e in range(echo):
            training_data_file = open("mnist_train.csv", 'r')
            training_data_list = training_data_file.readlines()
            training_data_file.close()
            index = 0
            for record in training_data_list:
                '''
                if index >= 1000:
                    break
                if you want to run it fast to see some output or optimize the model
                '''
               # print(record)
               # Use comma , to segment, every other , to segment a string element, each loop record will be updated once, updated to the latest training samples,
                all_values = record.split(',')

                inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                # inputs is a matrix contains every pixel
                #print(inputs)
               #Limit the data size range to 0.01-1.01, we want to avoid having 0 as the input value of the neural network, so here the +0.01 operation is performed
                targets = numpy.zeros(self.onodes) + 0.01
                targets[int(all_values[0])] = 0.99

                '''
                target is like [0.01 0.99 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01]
                it means it is num 1, we do it because target cannot be 0 and 1, it must be in (0, 1)
                '''
                self.train(inputs, targets)
            pass
        pass

# Set the number of neurons in each layer and the learning rate, note that the number of neurons in each layer is set here, not the number of layers, our program is just a simple 2-layer neural network model
# input layer has 784 neurons, which means we have a total of 784 data to feed to the neural network






