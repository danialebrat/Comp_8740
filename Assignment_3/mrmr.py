
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

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

################################################################
# %%
# Encode labels


le = LabelEncoder()
y = le.fit_transform(y)

print('number of samples and labels: {}'.format(y.shape))

################################################################
#%%
#Normalize samples' features


min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X)

################################################################
#%%
X = dataset.drop('Class',axis=1)
y = dataset['Class']

###############################################################
# DO NOT RUN THIS SECTION UNTIL YOU NEED TO RECACULATE THE RMRM FEATURE SELECTION,
# IT TAKES LONG TIMES TO RUN
#%% 
number_of_features_as_K = [500, 1000, 1500]
mrmr = {}
for i, k in enumerate(number_of_features_as_K):
    mrmr[str(k)] = mRMR_Feature_Selector.mrmr_classif(X = X, y=y, K=k)
# print('List of features selected by mRMR :', mrmr)

################################################################
#%% 
# saving mrmr1 features as a file for loading later
number_of_features_as_K = [500, 1000, 1500]
for _, k in enumerate(number_of_features_as_K):
    with open('mrmr_features_{}'.format(str(k)), 'wb') as fp:  # pickling
        pk.dump(mrmr[str(k)], fp)

################################################################
#%% 
# loading mrmr1 featers from the file
for _, k in enumerate(number_of_features_as_K):
    with open('mrmr_features_{}'.format(str(k)), 'rb') as fp:  # unpickling
        pk.dump(mrmr[str(k)], fp)

# with open('mrmr_features', 'rb') as fp:  # unpickling
#     mrmr1 = pk.load(fp)

################################################################
#%%
X_mrmr = {}
X_train = {}
X_test = {}
y_train = {}
y_test = {}
for _, k in enumerate(number_of_features_as_K):
    X_mrmr[str(k)] = X[mrmr[str(k)]]
    X_train[str(k)], X_test[str(k)], y_train[str(k)], y_test[str(k)] = train_test_split(X_mrmr[mrmr[str(k)]], 
                                                                                        y[mrmr[str(k)]], 
                                                                                        test_size = 0.2, 
                                                                                        random_state = 42)


print('type of targets', y_test[str(k)].unique())


################################################################
#%%
# model selection
clf = {}
for _, k in enumerate(number_of_features_as_K):
    clf = SVC(kernel = 'rbf', random_state = 42, decision_function_shape='ovr')
    clf.fit(X_train[str(k)], y_train[str(k)])

################################################################
#%%
y_pred = {}
cm = {}
for _, k in enumerate(number_of_features_as_K):
    y_pred[str(k)] = clf.predict(X_test[str(k)])
    # Confusion matrix
    cm[str(k)] = confusion_matrix(y_test[str(k)], y_pred[str(k)])
    
    print('confusion matrix for {} features =\n {}'.format(k, cm[str(k)]))


################################################################
#%%
# 10-folded cross validation and metric calculation
accuracies = {}
for _, k in enumerate(number_of_features_as_K):
    accuracies[str(k)] = cross_val_score(estimator = clf, X = X_train[str(k)], y = y_train[str(k)], cv = 10)
    print("Accuracy for {} features : {:.2f} %".format(k, accuracies[str(k)].mean()*100))
    print("Standard Deviation for {} features : {:.2f} %".format(k, accuracies[str(k)].std()*100))


################################################################
#%%
FP = cm.sum(axis=0) - np.diag(cm)  
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

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

print("PPV:{}\nNPV:{}\nSensitivity:{}\nSpecificity:{}".format(PPV, NPV, TPR, TNR))
print('\n')
print("Accuracy:",ACC)


avg_PPV  = np.average(PPV, axis=None, weights=None, returned=False)
avg_NPV  = np.average(NPV, axis=None, weights=None, returned=False)
avg_TPR  = np.average(TPR, axis=None, weights=None, returned=False)
avg_TNR  = np.average(TNR, axis=None, weights=None, returned=False)
avg_ACC  = np.average(ACC, axis=None, weights=None, returned=False)
print("PPV:{:.2f}\nNPV:{:.2f}\nSensitivity:{:.2f}\nSpecificity:{:.2f}\nAccuracy:{:.2f}".format(avg_PPV, avg_NPV, avg_TPR, avg_TNR, avg_ACC))


################################################################
#%%
#






# %%
