import matplotlib as mpl
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import number_recog
import numpy
import csv

#Setting up the model
input_nodes = 784
hidden_nodes = 100
# Note that this is the number of nodes in the hidden layer, not the number of layers in the hidden layer. Increase the number of nodes in the hidden layer, the code run time is significantly longer
output_nodes = 10
# output layer has 10 neurons, 0-9
learning_rate = 0.3
n = number_recog.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
n.pretrain()
print("Pretrain is done")

def find_gradient(all_values):
    gradient=numpy.zeros_like(numpy.asfarray(all_values[1:]))
    index=0
    targets = numpy.zeros(n.onodes) + 0.01
    targets[int(all_values[0])] = 0.99
    while index < numpy.size(gradient):
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        maximumLoss=n.lossFunction(inputs,targets)
        for c in range(0,255):
            temp = numpy.asfarray(all_values[1:])
            original=temp[index]
            temp[index]=c
            inputs = ((temp) / 255.0 * 0.99) + 0.01
            loss=n.lossFunction(inputs,targets)
            if loss>maximumLoss:
                gradient[index]=original-c
                maximumLoss=loss
        #print(str(index)+' : '+str(gradient[index]))
        index=index+1
    return gradient
    pass


def add_gradient(original, gradient, epsilon):
    r=numpy.asfarray(original)-(gradient*epsilon)
    return r
    pass

test_data_file = open("test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

start=int(input("Where to start: "))
test_data_list=test_data_list[start:]

# To create a csv of all input data into respective gradients
#epsilon= input("What is the epsilon value you would like to use: ")
with open('Gradient_1'+'.csv','a+',newline='') as f:
    writer=csv.writer(f)
    x=start
    for record in test_data_list:
        print("On "+str(x)+" image")
        all_values = record.split(',')
        grad=find_gradient(all_values)
        toWrite= list(grad)
        toWrite.insert(0, all_values[0])
        #print(toWrite)
        writer.writerow(toWrite)
        f.flush()
        x=x+1
'''
#Shows the original image, then the gradient, then the produced adversial image
epsilon=0.1
for i in range(0,3):
    record=test_data_list[i]
    all_values=record.split(',')
    #show Original
    image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()

    #original
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correct_label):
        print("Original was correct with", correct_label)
    else:
        print("Original was wrong with", correct_label, label)

    #show Gradient
    grad = find_gradient(all_values)
    grad_show=numpy.copy(grad)
    for i, value in enumerate(grad_show):
        if value<0:
            grad_show[i]=value*(-1)
    image_array = grad_show.reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()

    #show Adverisal
    adversial = add_gradient(all_values[1:],grad,epsilon)
    image_array = adversial.reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()

    #Adversial
    correct_label = int(all_values[0])
    inputs = ((adversial) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correct_label):
        print("Adversial was correct with", correct_label)
    else:
        print("Adversial was wrong with", correct_label, label)
'''
