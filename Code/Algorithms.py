from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd


def knnClassify(X, y, testData, coneValue, coneType, k):

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    result = knn.predict(testData)

    calcF1(result, coneValue, coneType)

    return result

def cluster(X, testData, coneValueTest, coneType):
    clustering = KMeans(n_clusters = 3)
    clustering.fit(X)

    predictions = [x + 1 for x in clustering.predict(testData)]   # have to add one, default is 0.0, 1.0, and 2.0

    calcF1(predictions, coneValueTest, coneType)

    return clustering

def pcaAnalysis(trainData):
    pca = PCA()
    pca.fit(trainData)
    pca_data = pca.transform(trainData)

    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

    pca_df = pd.DataFrame(pca_data, columns=labels)

    return per_var, labels, pca_df

def svmAnalysis(X, y, testData, coneValue, coneType):
    model = svm.SVC(gamma="scale")
    model.fit(X, y)

    result = model.predict(testData).tolist()
    print classification_report(coneValue, result)

    calcF1(result, coneValue, coneType)
    return result, model


def neuralNet(X, y, testData, coneValue, coneType):
    mlp = MLPClassifier(hidden_layer_sizes=(50,40,30,20,10))  # creates neural network instance
    mlp.fit(X, y)  # fits variables to NN

    # use model to predict
    predictions = mlp.predict(testData).tolist()

    print calcF1(predictions, coneValue, coneType)
    print classification_report(coneValue, predictions)



def calcF1(modelVal, trueVal, coneType):  # cone type can be 1.0, 2.0, or 3.0 depending on s, m, or l cone test
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    coneTypeCount = 0

    for i in range(len(modelVal)):
        if modelVal[i] == coneType:
            coneTypeCount += 1
            if trueVal.iloc[i] == coneType:
                tp += 1
            else:
                fp += 1
        elif modelVal[i] != coneType:
            if trueVal.iloc[i] != coneType:
                tn += 1
            elif trueVal.iloc[i] == coneType:
                fn += 1

    print "tp: " + str(tp)
    print "fp: " + str(fp)
    print "tn: " + str(tn)
    print "fn: " + str(fn)

    try:
        precision = tp / (tp + fp * 1.0)
        recall = tp / (fn + tp * 1.0)
        F1score = 2 * (precision * recall / (precision + recall * 1.0))   # F1 Score calculation
    except ZeroDivisionError:  # in case true positive is 0
        print "-Division by zero has occurred-"
        print "-Self-check results-"
        precision = 0
        recall = 0
        F1score = 0

    print "precision: " + str(precision)
    print "recall: " + str(recall)
    print "F1 score: " + str(F1score)

    try:
        print "\nModel's predicted amount: " + str(modelVal.count(coneType))
        print modelVal.count(2.0)
        print modelVal.count(1.0)
        print "Actual amount: "
        print str(trueVal.value_counts())
    except:
        pass
