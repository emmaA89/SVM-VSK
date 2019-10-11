""" Authors: C. Campi, F. Marchetti, E. Perracchione """
""" University of Padova """
""" Free software related the the paper (cite as): 
Variably scaled kernels as feature augmentation/extraction algorithms 
for classification: Fusing SVM and Naive Bayes, preprint 2019 """

""" Import the libraries """
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pylab as plt
import time

""" Fix the random seed """
seedo = 1992; np.random.seed(seedo); random.seed(seedo)

""" Download the data set """
# The dataset is available at 
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
db = np.loadtxt('breast-cancer-wisconsin.data',delimiter=',')
np.random.shuffle(db)

""" Define the data set after shuffle """
X_total = db[:,1:(np.size(db,axis=1)-1)]
y_total = db[:,(np.size(db,axis=1)-1):np.size(db,axis=1)]
y_total = np.array([-1 if yy == 2 else 1 for yy in y_total])
print('\n\n')
print('*******************************************************************')
print("Dataset uploaded")

""" The parameters used for the experiments """
divide = 2 # Ratio between the full dataset and the test set
folda = 5 # Number of folds for the cross validation
kernel = "linear" # The kernel can be linear or rbf
num_best_features = 2 # Number of best features for SVM (selected 
# via feature extraction)
num_prob_features = X_total.shape[1]-num_best_features+1 # The remining 
# feature used by NB 
excluded = 5 # Randomly selected feature
prob_features = [0,1,2,3,4,6,7,8] # Exclude num_best_features-1, here the 
# 5-th is excluded

""" The parameters used for cross-validation """
C_values = [2**i for i in range(-5,5)]
p_grid = { 
	"rbf" : {"C": C_values, "kernel": ["rbf"], "gamma" : 
        [10**i for i in range(-5, 5)]},
	"poly" : {"C": C_values, "kernel": ["poly"], "degree": 
        [1+i for i in range(4)]},
	"precomputed" : {"C": C_values, "kernel": ["precomputed"]},
    "linear" : {"C": C_values, "kernel": ["linear"]}
}

""" Divide the dataset for test and validation """
scaler = MinMaxScaler() # The scaling used in the tests
X_rest = X_total[X_total.shape[0]//divide:,:]
y_rest = y_total[y_total.shape[0]//divide:]
X = X_total[:X_total.shape[0]//divide,:]
y = y_total[:y_total.shape[0]//divide]
X_rest_ = X_rest.copy()
tt = time.time() # For computing the CPU time

""" Perform feature extraction """
skf = StratifiedKFold(n_splits=folda, shuffle=True, random_state=42)
fold, best_score = 1, 0
for train, test in skf.split(X, y):
    X_train, X_test = X[train],  X[test]
    
    """ Preprocessing """
    scaler.fit(X_train)
    X_train = scaler.transform(X_train.copy())
    X_test = scaler.transform(X_test.copy())
    X_rest = scaler.transform(X_rest_.copy())

    """ Grid search validation """
    clf = GridSearchCV(svm.SVC("linear"), param_grid=p_grid["linear"
                    ], cv=folda, scoring='f1_weighted') 
	
    """ Training """
    clf.fit(X_train, y[train].ravel())
	 
    """ Test """
    y_pred = clf.best_estimator_.predict(X_test)
	
    """ Compute the f1-score """
    acc = f1_score(y[test], y_pred,average = 'weighted') 
    
    """ Return the best performance among the different folds """
    if acc > best_score:
        best_score = acc
        best_modello = clf.best_estimator_

    fold += 1

""" Rank the features """
features_rank = np.argsort(np.abs(best_modello.coef_.ravel()))[::-1]

""" Define the selected feautures, reduce the dataset """
X_plot = X_rest.copy()
best_modello_plot = best_modello
X_rest = X_rest_[:,features_rank[:num_best_features]]
X = X[:,features_rank[:num_best_features]]
X_rest_ = X_rest.copy()

""" Perform SVM on the extracted features """
fold, accs, best_score = 1, [], 0
for train, test in skf.split(X, y):
    X_train, X_test = X[train],  X[test]
    
    """ Preprocessing """
    scaler.fit(X_train)
    X_train = scaler.transform(X_train.copy())
    X_test = scaler.transform(X_test.copy())
    X_rest = scaler.transform(X_rest_.copy())
    	
    """ Grid search validation """
    clf = GridSearchCV(svm.SVC(kernel), param_grid=p_grid[kernel], 
                       cv=folda, scoring='f1_weighted') 
	
    """ Training """
    clf.fit(X_train, y[train].ravel())
    
    """ Test """
    y_pred = clf.best_estimator_.predict(X_test)
    y_true = y[test]
    
    """ Compute the f1-score """
    acc = f1_score(y_true, y_pred,average = 'weighted') # Accuracy
    accs.append(acc)
    
    """ Return the best performance among the different folds """
    if acc > best_score:
        best_score = acc
        best_modello = clf.best_estimator_

    fold += 1

""" Compute the CPU time for feature extraction and SVM """
elapsed1 = time.time() - tt

""" Plot the feature ranking """
print('\n\n')
print('*******************************************************************')
print("FIGURE OF FEATURE RANKING")
fig, ax = plt.subplots()
plt.bar(np.arange(len(best_modello_plot.coef_.ravel())),
        best_modello_plot.coef_.ravel())
plt.xticks([i for i in range(0,0+np.size(X_plot,axis=1))])
plt.show()

""" Print the report on features """
print('\n\n')
print('*******************************************************************')
print("REPORT ON FEATURES")
print('\n')
print("FEATURES USED BY NB")
print(prob_features)
print('\n')
print("FEATURES RANDOMLY SELECTED FOR SVM-VSK")
print(excluded)
print('\n')
print("FEATURE EXTRACION REPORT")
print(features_rank)
print('\n')
print("FEATURESS USED BY SVM")
print(features_rank[:num_best_features])
print('*******************************************************************')

""" Print the results for feature extraction and SVM """
print('\n\n')
print('*******************************************************************')
print("TEST1 CLASSIFICATION REPORT: STANDARD SVM WITH SELECTED FEATURES")
y_pred_test = best_modello.predict(X_rest)
acc_test = f1_score(y_rest, y_pred_test,average = 'weighted')
print(metrics.classification_report(y_rest, y_pred_test))
print('\n')
print("TEST1 CONFUSION MATRIX: STANDARD SVM WITH SELECTED FEATURES")
print(metrics.confusion_matrix(y_rest, y_pred_test))
print('\n')
print("TEST1 F1 SCORE: STANDARD SVM WITH SELECTED FEATURES")
print(acc_test)
print('\n')
print("TEST1 CPU TIME: STANDARD SVM WITH SELECTED FEATURES")
print(elapsed1)
print('*******************************************************************')

""" Divide the data set for test and validation for SVM-VSK """
X_rest = X_total[X_total.shape[0]//divide:,:]
y_rest = y_total[y_total.shape[0]//divide:]
X = X_total[:X_total.shape[0]//divide,:]
y = y_total[:y_total.shape[0]//divide]
tt = time.time() #Evaluate the CPU time for SVM-VSK

""" Define the features for NB """
rest_of_features = list(set([i for i in range(0,np.size(X_total,axis=1))])
-set(prob_features))

""" Perform NB """
pos_a = np.where(y == 1)[0] # Preprocessing
pos_b = np.where(y == -1)[0] 

""" Estimate the probabilities """
pre_prob_a = len(pos_a)/len(y)
pre_prob_b = 1-pre_prob_a

""" Fit and test the NB classifier """
nb = GaussianNB()
nb.class_prior_ = [pre_prob_b,pre_prob_a]
nb.fit(X[:,prob_features],y)

""" Generate the extra feature for SVM-VSK """
vsk_add = np.flip(nb.predict_proba(X[:,prob_features]),axis=1)
vsk_add_rest = np.flip(nb.predict_proba(X_rest[:,prob_features]),axis=1)
X = X[:,rest_of_features]  
X_rest = X_rest[:,rest_of_features]
X = np.hstack((X.copy(),np.expand_dims(vsk_add[:,0],axis=1)))
X_rest = np.hstack((X_rest.copy(),np.expand_dims(vsk_add_rest[:,0],axis=1)))
X_rest_ = X_rest.copy()

""" Perform SVM-VSK on the extracted features """
fold, accs, best_score = 1, [], 0
for train, test in skf.split(X, y):
    X_train, X_test = X[train],  X[test]
    
    """ Preprocessing """
    add = X_train[:,-1:]
    add_ = X_test[:,-1:]
    add__ = X_rest[:,-1:]
    scaler.fit(X_train[:,:-1])
    X_train = np.hstack((scaler.transform(X_train[:,:-1].copy()),add))
    X_test = np.hstack((scaler.transform(X_test[:,:-1].copy()),add_))
    X_rest = np.hstack((scaler.transform(X_rest_[:,:-1].copy()),add__))  
    
    """ Grid search validation """
    clf = GridSearchCV(svm.SVC(kernel), param_grid=p_grid[kernel
                    ], cv=folda, scoring='f1_weighted') 
	
    """ Training """
    clf.fit(X_train, y[train].ravel())

    """ Test """	
    y_pred = clf.best_estimator_.predict(X_test)
    y_true = y[test]
	
    """ Compute the f1-score """
    acc = f1_score(y_true, y_pred,average = "weighted")
    accs.append(acc)
    
    """ Return the best performance among the different folds """
    if acc > best_score:
        best_score = acc
        best_modello = clf.best_estimator_
        X_rest_ok = X_rest.copy()

    fold += 1
    
""" Compute the CPU time for feature extraction and SVM """
elapsed2 = time.time() - tt

""" Print the results for SVM-VSK """
print('\n\n')
print('*******************************************************************')
print("TEST2 CLASSIFICATION REPORT: VSK-SVM")
y_pred_test = best_modello.predict(X_rest_ok)
acc_test = f1_score(y_rest, y_pred_test,average = 'weighted')
print(metrics.classification_report(y_rest, y_pred_test))
print('\n')
print("TEST2 CONFUSION MATRIX: VSK-SVM")
print(metrics.confusion_matrix(y_rest, y_pred_test))
print('\n')
print("TEST2 F1 SCORE: VSK-SVM")
print(acc_test)
print('\n')
print("TEST2 CPU TIME: VSK-SVM")
print(elapsed2)
print('*******************************************************************')