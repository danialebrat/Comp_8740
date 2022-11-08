
# this code is not running properly
#%%
import My_RMRM

import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
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

number_of_features_as_K = 500


X = dataset.drop('Class',axis=1)
y = dataset['Class']

################################################################
#%%

mrmr1 = My_RMRM.mrmr_classif(X = X, y=y, K=500)
print('Original number of features:', X.shape)
# print('Reduced number of features:', X_kbest.shape)

################################################################
#%%

X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size = 0.2, random_state = 42)
X_train_IG, X_test_IG, y_train_IG, y_test_IG = train_test_split(X_kbest_IG, y, test_size = 0.2, random_state = 42)
print(y_test.unique())
print(y_test_IG.unique())

# model selection

classifier = SVC(kernel = 'rbf', random_state = 42, decision_function_shape='ovr')
classifier.fit(X_train, y_train)
classifier_IG = SVC(kernel = 'rbf', random_state = 42, decision_function_shape='ovr')
classifier_IG.fit(X_train_IG, y_train_IG)

################################################################
#%%
# Confusion matrix

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

y_pred_IG = classifier.predict(X_test_IG)
cm_IG = confusion_matrix(y_test_IG, y_pred_IG)
print(cm_IG)

################################################################
#%%
# 10-folded cross validation and metric calculation

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

accuracie_IG = cross_val_score(estimator = classifier, X = X_train_IG, y = y_train_IG, cv = 10)
print("IG Accuracy: {:.2f} %".format(accuracie_IG.mean()*100))
print("IG Standard Deviation: {:.2f} %".format(accuracie_IG.std()*100))

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

FP_IG = cm_IG.sum(axis=0) - np.diag(cm_IG)  
FN_IG = cm_IG.sum(axis=1) - np.diag(cm_IG)
TP_IG = np.diag(cm_IG)
TN_IG = cm_IG.sum() - (FP + FN + TP)

print(FP_IG)
print(FN_IG)
print(TP_IG)
print(TN_IG)

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


# Sensitivity, hit rate, recall, or true positive rate
TPR_IG = TP_IG/(TP_IG+FN_IG)
# Specificity or true negative rate
TNR_IG = TN_IG/(TN_IG+FP_IG) 
# Precision or positive predictive value
PPV_IG = TP_IG/(TP+FP_IG)
# Negative predictive value
NPV_IG = TN_IG/(TN_IG+FN_IG)
# Fall out or false positive rate
FPR_IG = FP_IG/(FP_IG+TN_IG)
# False negative rate
FNR_IG = FN_IG/(TP_IG+FN_IG)
# False discovery rate
FDR_IG = FP_IG/(TP_IG+FP_IG)


# Overall accuracy
ACC_IG = (TP_IG+TN_IG)/(TP_IG+FP_IG+FN_IG+TN_IG)

print("PPV:{}\nNPV:{}\nSensitivity:{}\nSpecificity:{}".format(PPV, NPV, TPR, TNR))
print('\n')
print("Accuracy:",ACC)

print("\n\n\nPPV_IG:{}\nNPV_IG:{}\nSensitivity_IG:{}\nSpecificity_IG:{}".format(PPV_IG, NPV_IG, TPR_IG, TNR_IG))
print('\n')
print("Accuracy_IG:",ACC_IG)

avg_PPV  = np.average(PPV, axis=None, weights=None, returned=False)
avg_NPV  = np.average(NPV, axis=None, weights=None, returned=False)
avg_TPR  = np.average(TPR, axis=None, weights=None, returned=False)
avg_TNR  = np.average(TNR, axis=None, weights=None, returned=False)
avg_ACC  = np.average(ACC, axis=None, weights=None, returned=False)
print("PPV:{:.2f}\nNPV:{:.2f}\nSensitivity:{:.2f}\nSpecificity:{:.2f}\nAccuracy:{:.2f}".format(avg_PPV, avg_NPV, avg_TPR, avg_TNR, avg_ACC))

avg_PPV_IG  = np.average(PPV_IG, axis=None, weights=None, returned=False)
avg_NPV_IG  = np.average(NPV_IG, axis=None, weights=None, returned=False)
avg_TPR_IG  = np.average(TPR_IG, axis=None, weights=None, returned=False)
avg_TNR_IG  = np.average(TNR_IG, axis=None, weights=None, returned=False)
avg_ACC_IG  = np.average(ACC_IG, axis=None, weights=None, returned=False)
print("\nPPV:{:.2f}\nNPV:{:.2f}\nSensitivity:{:.2f}\nSpecificity:{:.2f}\nAccuracy:{:.2f}".format(avg_PPV_IG, avg_NPV_IG, avg_TPR_IG, avg_TNR_IG, avg_ACC_IG))

################################################################
#%%
#





