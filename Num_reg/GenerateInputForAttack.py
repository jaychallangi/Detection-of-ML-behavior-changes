import matplotlib as mpl
import matplotlib as mpl
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy
import csv

test_data_file = open("test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

grad_data_file = open("Gradient_1.csv", 'r')
grad_data_list = grad_data_file.readlines()
grad_data_file.close()


epsilon=0.1
with open('Epsilon_0.1_1'+'.csv','a+',newline='') as f:
    writer = csv.writer(f)
    for record,grad in zip(test_data_list,grad_data_list):
        all_values = record.split(',')
        all_grad = grad.split(',')
        toWrite=list(numpy.asfarray(all_values[1:])-(numpy.asfarray(all_grad[1:])*epsilon))
        toWrite.insert(0,all_values[0])
        writer.writerow(toWrite)
        f.flush()

'''
#Generating pictures we can see of the gradient and adversial
for e in [0.1,0.2,0.3]:
    for i in range(0,3):
        record=test_data_list[i]
        all_values=record.split(',')
        #show Original
        image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
        plt.imshow(image_array, cmap='Greys', interpolation='None')
        plt.show()

        #show Gradient
        record = grad_data_list[i]
        all_grad = record.split(',')
        grad_show = numpy.asfarray(all_grad[1:])
        #print(grad_show)
        for i, value in enumerate(grad_show):
            if value<0:
                grad_show[i]=value*(-1)
        #print(grad_show)
        image_array = grad_show.reshape((28, 28))
        plt.imshow(image_array, cmap='Greys', interpolation='None')
        plt.show()

        #show Adverisal
        adversial = numpy.asfarray(all_values[1:])-(numpy.asfarray(all_grad[1:])*e)
        image_array = adversial.reshape((28, 28))
        plt.imshow(image_array, cmap='Greys', interpolation='None')
        plt.show()
'''

