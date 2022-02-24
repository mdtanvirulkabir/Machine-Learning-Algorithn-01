# noinspection PyInterpreter
import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def cifar10_classifier_random(test_data):
    # The function produces the random labels for classify the images.
    test_data_dim = test_data.shape
    num_pic = test_data_dim[0] # The num_pic will have number of pics in test set.
    label_list = [] # This list will save the predicted labels from the classifier.
    for num in range(num_pic):
        label_list.append(np.random.randint(9))
    return label_list

def cifar10_classifier_1nn(test_data, tr_data, tr_level):
    # The function produces the 1 nearest labels for classify the images.
    test_data_dim = test_data.shape
    num_pic_t = test_data_dim[0]

    tr_data_dim = tr_data.shape
    num_pic_tr = tr_data_dim[0]
    predict_label_list = []
    for num_t in range(num_pic_t):
        distance_record = []
        pic_t = test_data[num_t]
        for num_tr in range(num_pic_tr):
            pic_tr = tr_data[num_tr]
            total_distance = np.sqrt(np.sum(np.abs(pic_tr - pic_t)))
            distance_record.append(total_distance)
        minimum_distance = min(distance_record)
        predict_pic_index = distance_record.index(minimum_distance)
        predict_label_list.append(tr_level[predict_pic_index])

    return predict_label_list


def class_acc(pred, ground_truth):
    # The function produces the classifier accuracy
    accurate_tab = 0
    for i in range(len(pred)):
        if pred[i] == ground_truth[i]:
            accurate_tab = 1 + accurate_tab
    accuracy = (accurate_tab*100) / len(ground_truth)
    return accuracy

def main():
    datadict = unpickle('E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week02/cifar-10-batches-py/test_batch')
    X = datadict["data"].astype("int32")
    Y = datadict["labels"]
    total_tr_data = []
    total_tr_labels = []

    for i in range(5):
        # Selecting the data batches and combining the all batches
        if i == 0:
            datadict_tr = unpickle('E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week02/cifar-10-batches-py/data_batch_1')

        elif i == 1:
            datadict_tr = unpickle('E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week02/cifar-10-batches-py/data_batch_2')

        elif i == 2:
            datadict_tr = unpickle(
                'E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week02/cifar-10-batches-py/data_batch_3')

        elif i == 3:
            datadict_tr = unpickle(
                'E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week02/cifar-10-batches-py/data_batch_4')

        elif i == 4:
            datadict_tr = unpickle(
                'E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week02/cifar-10-batches-py/data_batch_5')

        X_tr = datadict_tr["data"].astype("int32")
        Y_tr = datadict_tr["labels"]
        total_tr_data = np.append(total_tr_data, X_tr)
        total_tr_labels = np.append(total_tr_labels, Y_tr)


    predict_label_list_Rand = cifar10_classifier_random(X)
    predict_label_list_1nn = cifar10_classifier_1nn(X, X_tr, Y_tr)

    nn1_classifier_accuracy = class_acc(predict_label_list_1nn, Y)
    print('1nn_classifier_accuracy ', nn1_classifier_accuracy, '% for test batch')

    random_classifier_accuracy = class_acc(predict_label_list_Rand,Y)
    print('random_classifier_accuracy ', random_classifier_accuracy, '% for test batch')


    labeldict = unpickle('E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week02/cifar-10-batches-py/batches.meta')
    label_names = labeldict["label_names"]

    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y = np.array(Y)

    for i in range(X.shape[0]):
        # Show some images randomly
        if random() > 0.999:
            plt.figure(1);
            plt.clf()
            plt.imshow(X[i])
            plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
            plt.pause(1)

main()



