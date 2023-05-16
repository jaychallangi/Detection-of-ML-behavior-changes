import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.special
from pandas import DataFrame
import number_recog

input_nodes = 784
hidden_nodes = 100
# Note that this is the number of nodes in the hidden layer, not the number of layers in the hidden layer. Increase the number of nodes in the hidden layer, the code run time is significantly longer
output_nodes = 10
# output layer has 10 neurons, 0-9
learning_rate = 0.3
n = number_recog.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
n.pretrain()

test_data_file = open("Epsilon_0.1_1.csv", 'r') #Change to names of the input attacks
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []
# compare correct_lable and predict_label, cal the array
'''
example of  Precision, Recall and F1-score
TP, FN, FP 
is: correct label, res: model label 
TP: is 3， res is 3
FN：is 3， res is not 3
FP：is not 3， res is 3

hashmap[tp, fn, fp]
perc = tp/tp+fp
rec = tp/tp+fn
f1 = 2*pre*rec/perc + rec

'''
#i: number from 0-9,[0, 0, 0]: TP, FN, FP

hashmap = {i:[0, 0, 0] for i in range(10)}
confidence = [[0, 0] for i in range(10)]
for record in test_data_list:

    all_values = record.split(',')

    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    outputs = n.query(inputs)

    label = numpy.argmax(outputs)
     # find the max probability number

    if (label == correct_label):
        confidence[label][0] += max(outputs)
        confidence[label][1] += 1
        print("correct", correct_label)
        hashmap[correct_label][0] += 1
        scorecard.append(1)
    else:
        print("wrong",correct_label,label)
        hashmap[correct_label][1] += 1
        hashmap[label][2] += 1
        scorecard.append(0)
    pass
pass
performance = numpy.asarray(scorecard).sum() / numpy.asarray(scorecard).size

# cal and draw picture of Precision, Recall and F1-score
precision,recall, F1_score, confid = [],[],[],[]

for i in range(10):
    tp, fn, fp = hashmap[i]
    prec = tp/(tp+fp) if (tp+fp) != 0 else 1
    rec = tp/(tp+fn) if (tp+fn) != 0 else 1
    f1 =  (2 * prec * rec) / (prec + rec)
    precision.append(prec)
    recall.append(rec)
    F1_score.append(f1)
    cur_fid = confidence[i][0]/confidence[i][1] if confidence[i][1]!=0 else 1
    confid.append(cur_fid)
#cal the deviation of results
def deviation(target):
    mean = sum(target) / len(target)
    variance = sum([((x - mean) ** 2) for x in target]) / len(target)
    res = variance ** 0.5
    return res
dev_p, dev_r, dev_f, dev_c = deviation(precision),deviation(recall),deviation(F1_score),deviation(confid)
listnum = [i for i in range(10)]
data_P = DataFrame(precision,listnum)
data_R = DataFrame(recall,listnum)
data_F =  DataFrame(F1_score,listnum)
data_C =  DataFrame(confid,listnum)
data_P.plot(title = "precision", xticks = (numpy.arange(10)), xlabel = "number", ylim = (0, 1))
data_R.plot(title = "recall", xticks = (numpy.arange(10)), xlabel = "number", ylim = (0, 1))
data_F.plot(title = "F1_score", xticks = (numpy.arange(10)), xlabel = "number", ylim = (0, 1))
data_C.plot(title = "Confidence",xticks = (numpy.arange(10)), xlabel = "number", ylim = (0, 1))

overall = [sum(precision)/10,sum(recall)/10,sum(F1_score)/10, sum(confid)/10]
print("precision,recall,F1_score, confidence:", overall)




