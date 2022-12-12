
# this code is not running properly
#%%
import mRMR_Feature_Selector

import os
import numpy as np
import pandas as pd
import pickle as pk # for saving the list

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
# import matplotlib.pyplot as plt
################################################################
#%%
#Read the dataset
dataset = pd.read_csv(os.getcwd()+'/Datasets/Breastcancer.csv')
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

X = dataset.iloc[:, :-1]
y_ = dataset.iloc[:, -1]

################################################################
# %%
# Encode labels


le = LabelEncoder()
y = le.fit_transform(y_)

print('number of samples and labels: {}'.format(y.shape))

################################################################
#%%
X = dataset.drop('Class',axis=1)
y = dataset['Class']

###############################################################
#%% 
# DO NOT RUN THIS SECTION UNTIL YOU NEED TO RECACULATE THE RMRM FEATURE SELECTION,
# IT TAKES LONG TIMES TO RUN

number_of_features_as_K = [50, 80, 100, 150, 200]
mrmr = {}
for i, k in enumerate(number_of_features_as_K):
    mrmr[k] = mRMR_Feature_Selector.mrmr_classif(X = X, y=y, K=k)
    with open('mrmr_features_{}'.format(str(k)), 'wb') as fp:  # pickling
        pk.dump(mrmr[k], fp)
    fp.close()
# print('List of features selected by mRMR :', mrmr)

################################################################
#%% 
# saving mrmr1 features as a file for loading later
number_of_features_as_K = [50, 80, 100, 150, 200, 500, 1000, 1500]
for _, k in enumerate(number_of_features_as_K):
    with open('mrmr_features_{}'.format(str(k)), 'wb') as fp:  # pickling
        pk.dump(mrmr[k], fp)
    fp.close()

################################################################
#%% 
# Loading mrmr1 featers from the file
number_of_features_as_K = [50, 80, 100, 150, 200, 500, 1000, 1500]
mrmr = {}
for _, k in enumerate(number_of_features_as_K):
    with open('mrmr_features_{}'.format(str(k)), 'rb') as fp:  # unpickling
        mrmr[k] = pk.load(fp)
    fp.close()


# with open('mrmr_features', 'rb') as fp:  # unpickling
#     mrmr1 = pk.load(fp)

################################################################
#%%
X_mrmr = {}
X_train = {}
X_test = {}
y_train = {}
y_test = {}

# Normalize samples' features
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X)

for _, k in enumerate(number_of_features_as_K):
    X_mrmr[k] = X[mrmr[k]]
    # y = y_[mrmr[k]]

    X_train[k], X_test[k], y_train[k], y_test[k] = train_test_split(X_mrmr[k], y, test_size = 0.2, random_state = 42)

    print('number of samples in testset for k of {} is {}'.format(k, len(y_train[k])))


################################################################
#%%
# model selection
clf = {}
RF = {}
DT = {}
for _, k in enumerate(number_of_features_as_K):
    clf[k] = SVC(kernel = 'rbf', random_state = 42, decision_function_shape='ovr')
    clf[k].fit(X_train[k], y_train[k])
    RF[k] = RandomForestClassifier(criterion='entropy')
    RF[k].fit(X_train[k], y_train[k])
    DT[k] = RandomForestClassifier()
    DT[k].fit(X_train[k], y_train[k])

################################################################
#%%
y_pred = {}
y_pred_DT = {}
y_pred_RF = {}

cm = {}
cm_DT = {}
cm_RF = {}

for _, k in enumerate(number_of_features_as_K):
    y_pred[k] = clf[k].predict(X_test[k])
    y_pred_RF[k] = RF[k].predict(X_test[k])
    y_pred_DT[k] = RF[k].predict(X_test[k])
    # Confusion matrix
    cm[k] = confusion_matrix(y_test[k], y_pred[k])
    cm_RF[k] = confusion_matrix(y_test[k], y_pred[k])
    cm_DT[k] = confusion_matrix(y_test[k], y_pred[k])
    
    print('SVM_confusion matrix for {} features =\n {}'.format(k, cm[k]))
    print('RF_confusion matrix for {} features =\n {}'.format(k, cm_RF[k]))
    print('DT_confusion matrix for {} features =\n {}'.format(k, cm_DT[k]))


################################################################
#%%
# 10-folded cross validation and metric calculation
accuracies = {}
accuracies_RF = {}
accuracies_DT = {}
for _, k in enumerate(number_of_features_as_K):
    accuracies[k] = cross_val_score(estimator = clf[k], X = X_train[k], y = y_train[k], cv = 10)
    accuracies_RF[k] = cross_val_score(estimator = RF[k], X = X_train[k], y = y_train[k], cv = 10)
    accuracies_DT[k] = cross_val_score(estimator = RF[k], X = X_train[k], y = y_train[k], cv = 10)
    print("SVM_Accuracy using {} features : {:.2f} %".format(k, accuracies[k].mean()*100))
    print("RF_Accuracy using {} features : {:.2f} %".format(k, accuracies_RF[k].mean()*100))
    print("DT_Accuracy using {} features : {:.2f} %".format(k, accuracies_DT[k].mean()*100))

    # print("Standard Deviation for {} features : {:.2f} %".format(k, accuracies[k].std()*100))


################################################################
#%%
def other_measures(cm, k, classifier_ = 'SVM'):
    print(type(cm))
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    print('\n',"#"*20,' Measurments using {} features in {}'.format(k, classifier_)," ","#"*20,'\n')
    print('{} Measurement table:'.format(classifier_))
    print(FP)
    print(FN)
    print(TP)
    print(TN)


    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)


    print("{} ===> PPV:{}\nNPV:{}\nSensitivity:{}\nSpecificity:{}".format(classifier_, PPV, NPV, TPR, TNR))
    print('\n')
    print("{} ===> Accuracy:".format(classifier_, ACC))

    avg_PPV  = np.average(PPV, axis=None, weights=None, returned=False)
    avg_NPV  = np.average(NPV, axis=None, weights=None, returned=False)
    avg_TPR  = np.average(TPR, axis=None, weights=None, returned=False)
    avg_TNR  = np.average(TNR, axis=None, weights=None, returned=False)
    avg_ACC  = np.average(ACC, axis=None, weights=None, returned=False)
    print("{} ===> PPV:{:.2f}\nNPV:{:.2f}\nSensitivity:{:.2f}\nSpecificity:{:.2f}\nAccuracy:{:.2f}".format(classifier_, avg_PPV, avg_NPV, avg_TPR, avg_TNR, avg_ACC))
    print('\n')

for _, k in enumerate(number_of_features_as_K):
    other_measures(cm[k], k, 'SVM')
    other_measures(cm_RF[k], k, 'RF')
    other_measures(cm_DT[k], k, 'DT')
    
################################################################
#%%
#






# %%
