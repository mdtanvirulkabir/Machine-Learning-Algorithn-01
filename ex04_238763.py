
import tensorflow as tf
import pickle
import numpy as np
import keras
from keras.layers import convolutional, MaxPooling2D
from keras.layers import Dropout, Flatten



def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def class_acc(pred, ground_truth):
    accurate_tab = 0
    for i in range(len(pred)):
        if pred[i] == ground_truth[i]:
            accurate_tab = 1 + accurate_tab
    accuracy = (accurate_tab*100) / len(ground_truth)
    return accuracy

def normal_NN(Y_trn, Y_test, scaling_X_trn, scaling_X_test):
    model = keras.Sequential([
        keras.layers.Dense(250, input_dim=3072, activation='sigmoid'),  # neurons in the first hidden layer
        keras.layers.Dense(250, activation='sigmoid'),  # hidden layers
        keras.layers.Dense(10, activation='sigmoid')  # output 10 classes
    ])

    opt = keras.optimizers.SGD(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss= 'mse',
                  metrics=['mse'])

    Y_one_hot_tr = tf.keras.utils.to_categorical(Y_trn, num_classes=10)
    model.fit(scaling_X_trn, Y_one_hot_tr, epochs=200)
    predict_labels = model.predict(scaling_X_test) # colms = 10, rows = scaling_X_test no of rows
    predict_labels_int = np.argmax(predict_labels, axis=1).astype(int)
    result = class_acc(predict_labels_int, Y_test)
    print('Model Accuracy for Normal Neural Network ', result, '%')


def convolution_NN(Y_trn, Y_test, scaling_X_trn, scaling_X_test):
    scaling_X_trn = scaling_X_trn.reshape(10000, 32, 32, 3)
    scaling_X_test = scaling_X_test.reshape(10000, 32, 32, 3)
    model_conv = keras.Sequential([

        keras.layers.Conv2D(128, kernel_size=3, input_shape=(32, 32, 3), activation= 'relu'),  # input

        keras.layers.Conv2D(512, kernel_size=3, activation='relu'),  # hidden
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(512, kernel_size=3, activation= 'relu'),
        keras.layers.Conv2D(512, kernel_size=3, activation= 'relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),

        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(10, activation='sigmoid')  # output
    ])

    opt = keras.optimizers.Adam(learning_rate=0.01)

    model_conv.compile(optimizer=opt,
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    Y_one_hot_tr = tf.keras.utils.to_categorical(Y_trn, num_classes=10)
    Y_one_hot_test = tf.keras.utils.to_categorical(Y_test, num_classes=10)
    model_conv.fit(scaling_X_trn, Y_one_hot_tr, epochs=3, batch_size=64)
    result = model_conv.evaluate(scaling_X_test, Y_one_hot_test, verbose=1)
    print('Model Accuracy for Convolution Neural Network', result[1] * 100, '%')




def main():

    total_tr_data = np.zeros((0, 3072))
    total_tr_labels = np.zeros(0)

    for i in range(5):
        # Selecting the data batches and combining the all batches
        if i == 0:
            datadict_tr = unpickle(
                'E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week03/cifar-10-batches-py/data_batch_1')

        elif i == 1:
            datadict_tr = unpickle(
                'E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week03/cifar-10-batches-py/data_batch_2')

        elif i == 2:
            datadict_tr = unpickle(
                'E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week03/cifar-10-batches-py/data_batch_3')

        elif i == 3:
            datadict_tr = unpickle(
                'E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week03/cifar-10-batches-py/data_batch_4')

        elif i == 4:
            datadict_tr = unpickle(
                'E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week03/cifar-10-batches-py/data_batch_5')

        X_tr = datadict_tr["data"].astype("float64")
        Y_tr = datadict_tr["labels"]
        total_tr_data = np.vstack((total_tr_data, X_tr))
        total_tr_labels = np.concatenate((total_tr_labels, Y_tr))


    scaling_X_trn = total_tr_data/255.0



    datadict_test = unpickle('E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week04/cifar-10-batches-py/test_batch')

    X_test = datadict_test["data"].astype('float32')
    Y_test = datadict_test["labels"]
    scaling_X_test = X_test/255.0


    normal_NN(total_tr_labels, Y_test, scaling_X_trn, scaling_X_test)

    convolution_NN(Y_trn, Y_test, scaling_X_trn, scaling_X_test)



main()





