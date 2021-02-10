#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
#from sklearn.preprocessing import StandardScaler, MinMaxScale
from sklearn.utils import shuffle
from sklearn.base import clone

# set the random state for reproducibility
import numpy as np
np.random.seed(401)

classifiers = [SGDClassifier(),
               GaussianNB(),
               RandomForestClassifier(n_estimators=10, max_depth=5),
               MLPClassifier(alpha=0.05),
               AdaBoostClassifier()]

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    result = 0
    if np.sum(C) != 0:
        result = np.divide(np.trace(C), np.sum(C))
    return result


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    denominators = np.sum(C, axis=1)
    result = np.divide(np.diagonal(C), denominators)
    return result


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    denominators = np.sum(C, axis=0)
    result = np.divide(np.diagonal(C), denominators)
    return result


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    '''
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")

    iBest = 0
    acc_best = 0

    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for i, classifier_clone in enumerate(classifiers):
            print(i)
            classifier = clone(classifier_clone)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)

            accuracy_result = accuracy(conf_matrix)
            recall_result = recall(conf_matrix)
            precision_result = precision(conf_matrix)
            if accuracy_result > acc_best:
                acc_best = accuracy_result
                iBest = i

            classifier_name = str(classifier.__class__).split(".")[-1].replace(">", "").replace("\'", "")
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {accuracy_result:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recall_result]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precision_result]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

        # For each classifier, compute results and write the following output:
        #     outf.write(f'Results for {classifier_name}:\n')  # Classifier name
        #     outf.write(f'\tAccuracy: {accuracy:.4f}\n')
        #     outf.write(f'\tRecall: {[round(item, 4) for item in recall]}\n')
        #     outf.write(f'\tPrecision: {[round(item, 4) for item in precision]}\n')
        #     outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")

    X_1k = X_train[:1000]
    y_1k = y_train[:1000]
    data_increments = [1000, 5000, 10000, 15000, 20000]

    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        classifier = clone(classifiers[iBest])

        for num_train in data_increments:
            X = X_train[:num_train]
            y = y_train[:num_train]

            classifier.fit(X, y)
            y_pred = classifier.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            accuracy_result = accuracy(conf_matrix)
            outf.write(f'{num_train}: {accuracy_result:.4f}\n')

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")

    # q1
    k_feat = 5
    selector_k5 = SelectKBest(f_classif, k_feat)
    X_new = selector_k5.fit_transform(X_train, y_train)
    features_32k = selector_k5.get_support(True) # q4
    p_values = selector_k5.pvalues_

    k_feat50 = 50
    selector_k50 = SelectKBest(f_classif, k_feat50)
    X_new2 = selector_k50.fit_transform(X_train, y_train)
    p_values_k50 = selector_k50.pvalues_

    # q2
    classifier = clone(classifiers[i])
    classifier.fit(X_new, y_train)
    y_pred = classifier.predict(selector_k5.transform(X_test))
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy_full = accuracy(conf_matrix)

    # q3
    k_feat_1k = 5
    selector_1k = SelectKBest(f_classif, k_feat_1k)
    X_new_1k = selector_1k.fit_transform(X_1k, y_1k)
    classifier_1k = clone(classifiers[i])
    classifier_1k.fit(X_new_1k, y_1k)
    y_pred_1k = classifier_1k.predict(selector_1k.transform(X_test))
    conf_matrix_1k = confusion_matrix(y_test, y_pred_1k)
    accuracy_1k = accuracy(conf_matrix_1k)

    # q4
    features_1k = selector_1k.get_support(True)
    feature_intersection = np.intersect1d(features_1k, features_32k,
                                  return_indices=True)
    top_5 = features_32k


    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.

        # for each number of features k_feat, write the p-values for
        # that number of features:
        outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')
        outf.write(f'{k_feat50} p-values: {[round(pval, 4) for pval in p_values_k50]}\n')

        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {top_5}\n')


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")

    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    kf = KFold(n_splits=5, shuffle=True)

    p_values = []
    all_kfold_accuracies = []
    for j, classifier_clone in enumerate(classifiers):
        fold = 0
        kfold_accuracies = []
        for train_index, test_index in kf.split(X_all):
            classifier = clone(classifier_clone)
            X_train, X_test = X_all[train_index], X_all[test_index]
            y_train, y_test = y_all[train_index], y_all[test_index]
            classifier.fit(X_train, y_train)
            C = confusion_matrix(y_test, classifier.predict(X_test))

            kfold_accuracies.append(accuracy(C))
            fold += 1
        all_kfold_accuracies.append(kfold_accuracies)

    for j in range(5):
        if j!=i:
            S, pvalue = ttest_rel(all_kfold_accuracies[j], all_kfold_accuracies[i])
            p_values.append(pvalue)



    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        for kf_acc in all_kfold_accuracies:
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kf_acc]}\n')
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # TODO: load data and split into train and test.
    # TODO : complete each classification experiment, in sequence.

    data = np.load(args.input)
    features_array = data.f.arr_0

    features = features_array[:, :173]
    labels = features_array[:, 173]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    X_train, y_train = shuffle(X_train, y_train)

    # 3.1
    print("Processing 3.1")
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)

    # 3.2
    print("Processing 3.2")
    (X_1k, y_1k) = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)

    # 3.3
    print("Processing 3.3")
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)

    # 3.4
    print("Processing 3.4")
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)

