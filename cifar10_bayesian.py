import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.stats import norm
import scipy

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def cifar_10_color(X_trn):
    Xf = np.zeros((0, 3))
    for row in X_trn:
        image_split = np.array(row)
        image_red = image_split[0:1024]
        image_green = image_split[1024:2048]
        image_blue = image_split[2048:3072]
        mean_r = np.mean(image_red)
        mean_g = np.mean(image_green)
        mean_b = np.mean(image_blue)
        image_mean = np.array([mean_r, mean_g, mean_b])
        Xf = np.vstack((Xf,image_mean))
    return Xf

def cifar_10_custom_window_color_N(X, k):
    X = X.reshape(len(X), 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    size = np.size(X, axis= 0)

    X_mean_test_window = np.zeros((size,3*k*k))
    for i in range(X.shape[0]):
        # Convert images to mean values of each color channel
        img = X[i]
        #img_8x8 = resize(img, (8, 8))
        img_2x2 = resize(img, (k, k))
        r_vals = img_2x2[:,:,0].reshape(k*k)
        g_vals = img_2x2[:,:,1].reshape(k*k)
        b_vals = img_2x2[:,:,2].reshape(k*k)
        n_image = np.concatenate((r_vals, g_vals, b_vals))
        X_mean_test_window[i, :] = n_image

    return X_mean_test_window

def cifar_10_test(X):

    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    X_mean = np.zeros((10000, 3))
    for i in range(X.shape[0]):
        # Convert images to mean values of each color channel
        img = X[i]
        img_8x8 = resize(img, (8, 8))
        img_1x1 = resize(img, (1, 1))
        r_vals = img_1x1[:,:,0].reshape(1*1)
        g_vals = img_1x1[:,:,1].reshape(1*1)
        b_vals = img_1x1[:,:,2].reshape(1*1)
        mu_r = r_vals.mean()
        mu_g = g_vals.mean()
        mu_b = b_vals.mean()
        X_mean[i, :] = (mu_r, mu_g, mu_b)

    return X_mean


def class_mean(data_class):
    colms = np.size(data_class[0], axis= 1)
    class_m = np.zeros((0, colms))
    for index in range(10):
        one_class = data_class[index]
        image_color_mean = []
        for color in range(colms):
            color_mean = np.mean(one_class[:, color])
            image_color_mean.append(color_mean)
        class_m = np.vstack((class_m, image_color_mean))

    return class_m


def class_sd(data_class):
    colms = np.size(data_class[0], axis=1)
    class_v = np.zeros((0, colms))
    for index in range(10):
        one_class = data_class[index]
        image_color_sd = []
        for color in range(colms):
            color_v = np.std(one_class[:, color])
            image_color_sd.append(color_v)
        class_v = np.vstack((class_v, image_color_sd))

    return class_v


def class_covar(data_class):
    colms = np.size(data_class[0], axis=1)
    full_class_covar = np.zeros(shape=(10, colms, colms))
    for index in range(10):
        full_class_covar[index, :, :] = np.cov(data_class[index], rowvar = False)

    return full_class_covar


def creating_class(Xf,Y_trn):
    colms = np.size(Xf, axis=1)
    X_class_00 = np.zeros((0, colms)).astype('float64')
    X_class_01 = np.zeros((0, colms)).astype('float64')
    X_class_02 = np.zeros((0, colms)).astype('float64')
    X_class_03 = np.zeros((0, colms)).astype('float64')
    X_class_04 = np.zeros((0, colms)).astype('float64')
    X_class_05 = np.zeros((0, colms)).astype('float64')
    X_class_06 = np.zeros((0, colms)).astype('float64')
    X_class_07 = np.zeros((0, colms)).astype('float64')
    X_class_08 = np.zeros((0, colms)).astype('float64')
    X_class_09 = np.zeros((0, colms)).astype('float64')

    for index in range(len(Y_trn)):
        if Y_trn[index] == 0:
            X_class_00 = np.vstack((X_class_00, Xf[index]))
        elif Y_trn[index] == 1:
            X_class_01 = np.vstack((X_class_01, Xf[index]))
        elif Y_trn[index] == 2:
            X_class_02 = np.vstack((X_class_02, Xf[index]))
        elif Y_trn[index] == 3:
            X_class_03 = np.vstack((X_class_03, Xf[index]))
        elif Y_trn[index] == 4:
            X_class_04 = np.vstack((X_class_04, Xf[index]))
        elif Y_trn[index] == 5:
            X_class_05 = np.vstack((X_class_05, Xf[index]))
        elif Y_trn[index] == 6:
            X_class_06 = np.vstack((X_class_06, Xf[index]))
        elif Y_trn[index] == 7:
            X_class_07 = np.vstack((X_class_07, Xf[index]))
        elif Y_trn[index] == 8:
            X_class_08 = np.vstack((X_class_08, Xf[index]))
        elif Y_trn[index] == 9:
            X_class_09 = np.vstack((X_class_09, Xf[index]))


    data_class = np.array([X_class_00, X_class_01, X_class_02, X_class_03, X_class_04, X_class_05, X_class_06, X_class_07, X_class_08, X_class_09])
    return data_class

def cifar_10_naivebayes_learn(Xf,Y_trn):
    data_class = creating_class(Xf, Y_trn)
    mu = class_mean(data_class)
    sigma = class_sd(data_class)
    p = np.array([[0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])
    return mu, sigma, p


def cifar_10_bayes_learn(Xf,Y_trn):
    data_class = creating_class(Xf, Y_trn)
    mu = class_mean(data_class)
    full_class_covar = class_covar(data_class)
    p = np.array([[0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])
    return mu, full_class_covar, p

def cifar_10_classifier_naivebayes(x,mu,sigma,p):
    List_probability_all_class = []
    for class_no in range(10):
        cls_prob = 1.0
        mean_row = mu[class_no]
        sd_row = sigma[class_no]
        for clr_channel in range(3):
            mean = mean_row[clr_channel]
            sd = sd_row[clr_channel]
            x_chennel = x[clr_channel]
            chennel_prob_distribution = norm(mean, sd)
            chennel_prob = chennel_prob_distribution.pdf(x_chennel)
            cls_prob = cls_prob*chennel_prob
        List_probability_all_class.append(cls_prob*0.1)
    High_prob_value = max(List_probability_all_class)
    index = List_probability_all_class.index(High_prob_value)
    return index

def cifar10_classifier_bayes(x,mu,sigma,p):
    List_probability_all_class = []
    for class_no in range(10):
        class_prob_distribution = scipy.stats.multivariate_normal(mu[class_no], sigma[class_no])
        class_prob = class_prob_distribution.logpdf(x)
        List_probability_all_class.append(class_prob)
    High_prob_value = max(List_probability_all_class)
    index = List_probability_all_class.index(High_prob_value)
    # print(index)
    return index


def class_acc(pred, ground_truth):
    accurate_tab = 0
    for i in range(len(pred)):
        if pred[i] == ground_truth[i]:
            accurate_tab = 1 + accurate_tab
    accuracy = (accurate_tab*100) / len(ground_truth)
    return accuracy

def plot_result(accuracy_list):
    plot_accuracy = plt.figure(figsize=(10, 5))
    plt.plot(accuracy_list)
    plt.xlabel("Window Size")
    plt.ylabel("Accuracy(%)")
    plot_accuracy.savefig('accuracy plot Ex 03', bbox_inches = 'tight', dpi=150)
    plt.show()


def main():
    total_tr_data = np.zeros((0, 3072))
    total_tr_labels = np.zeros(0)

    for i in range(5):
        # Selecting the data batches and combining the all batches
        if i == 0:
            datadict_tr = unpickle('E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week03/cifar-10-batches-py/data_batch_1')

        elif i == 1:
            datadict_tr = unpickle('E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week03/cifar-10-batches-py/data_batch_2')

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



    datadict = unpickle('E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week03/cifar-10-batches-py/test_batch')

    X = datadict["data"].astype("float")
    Y = datadict["labels"]
    Y = np.array(Y)


    labeldict = unpickle('E:/Tau-master-BME/Introduction_Machine_Learning_and_pattern_recognition/Lecture and ex/week03/cifar-10-batches-py/batches.meta')
    label_names = labeldict["label_names"]

    window_size = [1, 2, 4, 8, 16]
    Gross_accuracy_list = []
    for window in window_size:

        if window == 1:
            Xf = cifar_10_color(total_tr_data)
            mu_naive, sigma, p_naive = cifar_10_naivebayes_learn(Xf, total_tr_labels)
            mu_bayes, full_class_covar, p_bayes = cifar_10_bayes_learn(Xf, total_tr_labels)
            predict_data_label_naivebayes = []
            predict_data_label_bayes = []
            X_mean = cifar_10_test(X)

            for image in X_mean:
                optimal_class_naivebayes = cifar_10_classifier_naivebayes(image, mu_naive, sigma, p_naive)
                optimal_class_bayes = cifar10_classifier_bayes(image, mu_bayes, full_class_covar, p_bayes)
                predict_data_label_naivebayes.append(optimal_class_naivebayes)
                predict_data_label_bayes.append(optimal_class_bayes)

            accuracy_naivebayes = class_acc(predict_data_label_naivebayes, Y)
            accuracy_bayes = class_acc(predict_data_label_bayes, Y)
            print('Naive Bayesian classifier accuracy for Test data label', accuracy_naivebayes, '%')
            print('Bayesian classifier for 1X1 window accuracy for Test data label', accuracy_bayes, '%')
            Gross_accuracy_list.append(accuracy_bayes)

        else:
            Xf_custom_window = cifar_10_custom_window_color_N(total_tr_data, window)
            Xf_custom_window_test = cifar_10_custom_window_color_N(X, window)
    # Calling Learing function for different classifier
            mu_bayes_kxk, full_class_covar_kxk, p_bayes_kxk = cifar_10_bayes_learn(Xf_custom_window, total_tr_labels)
            predict_data_label_bayes_kxk = []
            for image in Xf_custom_window_test:
                optimal_class_bayes_kXk = cifar10_classifier_bayes(image, mu_bayes_kxk, full_class_covar_kxk, p_bayes_kxk)
                predict_data_label_bayes_kxk.append(optimal_class_bayes_kXk)
            accuracy_bayes_kxk = class_acc(predict_data_label_bayes_kxk, Y)
            print('Bayesian classifier with', window, 'X', window, 'window accuracy for Test data label', accuracy_bayes_kxk, '%')
            Gross_accuracy_list.append(accuracy_bayes_kxk)

    plot_result(Gross_accuracy_list) # accuracy for each window will store in the plot_result list

    if random() > 0.999:
        plt.figure(1);
        plt.clf()
        plt.imshow(img_8x8)
        plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        plt.pause(1)



main()


#print("After Reshape", X)
#print("After Reshape certain portion", X[1000, 3, 32, 32])
"""
    """





    # Show some images randomly
"""
    
        """


#print(predict_data_label)




    


