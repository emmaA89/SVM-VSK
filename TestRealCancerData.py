""" Authors: C. Campi, F. Marchetti, E. Perracchione """

""" Free software related the the paper (cite as): 
Learning via variably scaled kernels """

# Import the needed libraries
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import numpy as np
import warnings

np.random.seed(1992) # Fix the random seed
warnings.filterwarnings('ignore')

###############################################################################
#### PARAMETERS

# Define the number of best features to select for SVM-S

num_best_features = 2

# Define the random seed for choosing the extracted features for SVM-E

extraction_seed = 42

# Define the kernel

kernel = "linear"

##############################################################################
#### DATASET CONSTRUCTION

# Load and divide the dataset into training and test set

db = np.loadtxt('breast-cancer-wisconsin.data',delimiter=',')
np.random.shuffle(db)

X_total = db[:,1:(np.size(db,axis=1)-1)]
y_total = db[:,(np.size(db,axis=1)-1):np.size(db,axis=1)]

y_total = np.array([-1 if yy == 2 else 1 for yy in y_total])

X_test = X_total[:X_total.shape[0]//2,:]
y_test = y_total[:y_total.shape[0]//2]
X_train = X_total[X_total.shape[0]//2:,:]
y_train = y_total[y_total.shape[0]//2:]

###############################################################################
#### CV PARAMETERS' GRID

C_values = [2**i for i in range(-6,6)]
gamma_values = [10**i for i in range(-6, 2)]

p_grid = { 
	"rbf" : {"C": C_values, "kernel": ["rbf"], "gamma" : gamma_values},
    "linear" : {"C": C_values, "kernel": ["linear"]}
}

###############################################################################
#### NAIVE BAYES CLASSIFIER

nb = GaussianNB()
nb.fit(X_train,y_train)

y_pred = nb.predict(X_test)

print('\n\n')

print("CLASSIFICATION REPORT: STANDARD NB")
	
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

print("TEST F1-SCORE NB:", np.around(f1_score(y_test, y_pred,\
                                            average = 'weighted'),3))

###############################################################################
#### STANDARD SVM

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = GridSearchCV(svm.SVC(kernel), param_grid=p_grid[kernel
                ], cv=5, scoring="f1_weighted") 
	
clf.fit(X_train, y_train.ravel())
y_pred = clf.predict(X_test)

print('\n\n')
print("-------------------------------------------------------------")
print("CLASSIFICATION REPORT: STANDARD SVM")
	
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

print("TEST F1-SCORE SVM:", np.around(f1_score(y_test, y_pred,\
                                            average = 'weighted'),3))

###############################################################################
#### SVM-S

# If the kernel is linear, we use the already existing model to perform
# the feature selection. Else, we train a linear SVM to get the coefficients

if kernel == "linear":
    
    # The most relevant features are sorted starting from the most important
    
    features_rank = np.argsort(np.abs(clf.best_estimator_.\
                                      coef_.ravel()))[::-1]
    
else:
    clf = GridSearchCV(svm.SVC("linear"), param_grid=p_grid[kernel
                ], cv=5, scoring="f1_weighted") 
	
    clf.fit(X_train, y_train.ravel())
    features_rank = np.argsort(np.abs(clf.best_estimator_.\
                                      coef_.ravel()))[::-1]

# Restrict to the best features in the datasets according to 
# num_best_features and features_rank

X_train = X_train[:,features_rank[:num_best_features]]
X_test = X_test[:,features_rank[:num_best_features]]

# Train and test SVM-S

clf = GridSearchCV(svm.SVC(kernel), param_grid=p_grid[kernel
                ], cv=5, scoring="f1_weighted") 

clf.fit(X_train, y_train.ravel())
y_pred = clf.predict(X_test)

print('\n\n')
print("-------------------------------------------------------------")
print("TEST CLASSIFICATION REPORT: SVM-S")
print("Considered best features:", features_rank[:num_best_features])
	
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

print("TEST F1-SCORE SVM-S:", np.around(f1_score(y_test, y_pred,\
                                            average = 'weighted'),3))

###############################################################################
#### SVM-E

# Recover the initial datasets and normalize them

X_train = scaler.fit_transform(X_total[X_total.shape[0]//2:,:])
y_train = y_total[y_total.shape[0]//2:]
X_test = scaler.transform(X_total[:X_total.shape[0]//2,:])
y_test = y_total[:y_total.shape[0]//2]

# Random selection of the compressed features for SVM-E

np.random.seed(42)
no_prob_features = np.sort(np.random.choice(X_train.shape[1],\
                                num_best_features-1,replace = False))

prob_features = list(set([i for i in range(0,X_train.shape[1])])-\
                     set(no_prob_features))
    
nb = GaussianNB()
nb.fit(X_train[:,prob_features],y_train)

add = np.flip(nb.predict_proba(X_train[:,prob_features]),axis=1)
add_test = np.flip(nb.predict_proba(X_test[:,prob_features]),axis=1)

# The new datasets are obtained by glueing the compressed feature to the
# remaining ones

X_train = np.hstack((X_train[:,no_prob_features],\
                      np.expand_dims(add[:,0],axis=1)))
X_test = np.hstack((X_test[:,no_prob_features],\
                    np.expand_dims(add_test[:,0],axis=1)))

# Train and test SVM-E

clf = GridSearchCV(svm.SVC(kernel), param_grid=p_grid[kernel],\
                    cv=5, scoring="f1_weighted") 
 	
clf.fit(X_train, y_train.ravel())
y_pred = clf.predict(X_test)
    
print('\n\n')
print("-------------------------------------------------------------")
print("TEST CLASSIFICATION REPORT: SVM-E")
print("Features compressed by NB:", prob_features)

print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

print("TEST F1-SCORE SVM-E:", np.around(f1_score(y_test, y_pred,\
                                            average = 'weighted'),3))