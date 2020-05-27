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
import random
import warnings

seedo = 1992  # Fix the random seed
warnings.filterwarnings('ignore') 

###############################################################################
#### PARAMETERS

# Fix the number of data among the (5000,64) dataset

num_data = 500

# Fix the number of features among the (5000,64) dataset

num_features = 2

# Define the kernel

kernel = "linear"

###############################################################################
#### CONSTRUCTION OF THE (5000,64) DATASET

np.random.seed(seedo)

# Define means and stds for both classes

mean_a  = np.random.uniform(0,20,size=5000)
std_a = np.random.uniform(0,2,size=5000)
mean_b  = mean_a.copy() + np.random.uniform(0,2,size=5000)
std_b = np.random.uniform(0,4.5,size=5000)

# Construct the label vector, i.e. the output

y_total = np.empty((5000,))
y_total[:2500]=1
y_total[2500:]=-1

# Construct the data matrix sampled from a normal distribution with the
# defined means and stds

X_a = np.empty((2500,64))

for i in range(0,np.size(X_a,axis=0)):
    for j in range(0,np.size(X_a,axis=1)):
        X_a[i,j] = np.random.normal(mean_a[j],std_a[j])
        
X_b = np.empty((2500,64))

for i in range(0,np.size(X_b,axis=0)):
    for j in range(0,np.size(X_b,axis=1)):
        X_b[i,j] = np.random.normal(mean_b[j],std_b[j])
        
# The final dataset is obtained by glueing X_a and X_b and then by adding a
# Gaussian white noise

X_total= np.vstack((X_a,X_b)) + np.random.normal(0,1,(5000,64))
    
##############################################################################
# DATASET EXTRACTION

# Sort the indices of the selected features and data, in order to obtain
# the (num_data,num_features) dataset. This is done preserving the balance
# between the two classes

np.random.seed(seedo)
random.seed(seedo)

selected_features = np.sort(np.random.choice(range(0,np.size(X_total,\
                                    axis=1)),num_features,replace = False))
selected_data = np.hstack((np.sort(random.sample(range(0,int(np.size(X_total,\
                axis=0)/2)),int(num_data/2))),np.sort(random.sample(range(int\
                (np.size(X_total,axis=0)/2),int(np.size(X_total,axis=0))),int\
                (num_data/2)))))

# Reduce the initial dataset according to the selected features

X_total = X_total[:,selected_features]
                                           
# Extract the (num_data,num_features) dataset                              

X = X_total[selected_data,:]
y = y_total[selected_data]

# Take the dataset with the data that have not been selected
       
X_excluded = np.delete(X_total,selected_data,axis=0)
y_excluded = np.delete(y_total,selected_data)

# Divide X,y into training and test sets (proportion 2:1) preserving
# the balance between the classes. We do this randomly selecting the test
# indices

test_indices = np.hstack((np.sort(random.sample(range(0,int(np.size(X,\
                axis=0)/2)),np.size(X,axis=0)//6)),np.sort(random.sample(\
                range(int(np.size(X,axis=0)/2),int(np.size(X,axis=0))),\
                np.size(X,axis=0)//6))))

X_test = X[test_indices,:]
y_test = y[test_indices]

X_train = np.delete(X,test_indices,axis=0)
y_train = np.delete(y,test_indices)

##############################################################################
#### CV PARAMETERS' GRID

C_values = [2**i for i in range(-6,6)]
gamma_values = [10**i for i in range(-6, 2)]

p_grid = { 
	"rbf" : {"C": C_values, "kernel": ["rbf"], "gamma" : gamma_values},
    "linear" : {"C": C_values, "kernel": ["linear"]}
}
#######################################################################
#### NAIVE BAYES CLASSIFIER

# NB is trained on the training dataset and on the data previously excluded
# from the extraction, in order to simulate a priori knowledge for
# SVM-VSK 
        
X_nb = np.vstack((X_excluded,X_train))
y_nb = np.hstack((y_excluded,y_train))
nb = GaussianNB()
nb.fit(X_nb,y_nb)

# We obtain the probabilistic output for X_train and X_test

nb_pred = np.flip(nb.predict_proba(X_train),axis=1)
nb_pred_test = np.flip(nb.predict_proba(X_test),axis=1)

# Test the NB classifier on the test dataset

bayes_class_test = np.argmax(nb_pred_test, axis=1)
bayes_class_test[bayes_class_test==1]=-1
bayes_class_test[bayes_class_test==0]=1

print("\n\n")

print("CLASSIFICATION REPORT: STANDARD NB")

print(metrics.classification_report(y_test, bayes_class_test))
print(metrics.confusion_matrix(y_test, bayes_class_test))
    
print("TEST F1-SCORE NB:", np.around(f1_score(y_test, bayes_class_test,\
                                     average = "weighted" ),3))

###############################################################################
#### STANDARD SVM

# Normalize the data
    
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)        

# Apply 5-fold CV on the training dataset  

clf = GridSearchCV(svm.SVC(kernel), param_grid=p_grid[kernel
                ], cv=5, scoring='f1_weighted') 
clf.fit(X_train,y_train)

# Test the constructed classifier on the test set

y_pred_test = clf.predict(X_test)

print("\n\n")
print("-------------------------------------------------------------")
print("CLASSIFICATION REPORT: STANDARD SVM")

print(metrics.classification_report(y_test, y_pred_test))
print(metrics.confusion_matrix(y_test, y_pred_test))

print("TEST F1-SCORE SVM:", np.around(f1_score(y_test, y_pred_test,\
                                     average = "weighted"),3))

###############################################################################
#### SVM-VSK

# Augment both training and test sets according to the output of NB
   
X_train = np.hstack((X_train,np.expand_dims(nb_pred[:,0],axis=1)))
X_test = np.hstack((X_test,np.expand_dims(nb_pred_test[:,0],axis=1)))

# Apply 5-fold CV on the augmented training dataset  

clf = GridSearchCV(svm.SVC(kernel), param_grid=p_grid[kernel
                ], cv=5, scoring='f1_weighted') 

clf.fit(X_train,y_train)

# Test the constructed VSK classifier on the test set

y_pred_test_vsk = clf.predict(X_test)

print("\n\n")
print("-------------------------------------------------------------")
print("CLASSIFICATION REPORT: SVM-VSK")

print(metrics.classification_report(y_test, y_pred_test_vsk))
print(metrics.confusion_matrix(y_test, y_pred_test_vsk))

print("TEST F1-SCORE SVM-VSK:", np.around(f1_score(y_test, y_pred_test_vsk,\
                                             average = "weighted"),3))


print("\n\n")
