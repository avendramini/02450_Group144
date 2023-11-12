# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:45:20 2023

@author: egk
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn import model_selection, tree
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy import stats
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the data set
filename = './Data/Data_for_project.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
attributeNames = df.columns[1:-2].tolist()

# Extract vector y, convert to NumPy array
raw_data = df.values  

column_to_predict = 10

# Seleziona tutte le colonne tranne quella da escludere
columns_for_prediction = [i for i in range(raw_data.shape[1]) if i != column_to_predict]

X=raw_data[:,columns_for_prediction]

y=raw_data[:,column_to_predict]
y=y.astype(float)
N=X.shape[0]
M=X.shape[1]

#Change present/not present to binary
for i in range(0,N):
    X[i][5]= 1.0 if X[i][5]=="Present" else 0.0

X=X.astype(float)

ones = np.ones((N,1),float)

X = X[:,1:]
print(X)
#K-fold cross validation
#Outer loop
K1=10
#inner loop 
K2=10

CV1 = model_selection.KFold(n_splits=K1, shuffle=True)

z_score = stats.zscore(X)

#Should I add the column of ones?
X = np.hstack((ones, z_score))

#Initialize variable
logistic_test_error = np.zeros(K1)
nbayes_test_error = np.zeros(K1)
baseline_test_error = np.zeros(K1)

k1=0
for par_index, test_index in CV1.split(X):
    print('Computing CV1 fold: {0}/{1}..'.format(k1+1,K1))
    print()
    
    #Find training and test set for current CV fold
    X_par, y_par = X[par_index, :], y[par_index]
    X_test, y_test = X[test_index,:], y[test_index]
    
    CV2 = model_selection.KFold(n_splits=K2, shuffle=False)
    
#Find lambda for logistic regression

    lambda_int = np.power(10., range(-5,9))
    logreg_generalisation_error_rate = np.zeros(len(lambda_int))
    for i in range(0, len(lambda_int)):
        k2 = 0
        logreg_valid_errorrate = np.zeros(K2)
        
        for train_index, val_index, in CV2.split(X_par):
            # extract training and test set for current CV fold
           X_train, y_train = X_par[train_index,:], y_par[train_index]
          
           
           X_val, y_val = X_par[val_index,:], y_par[val_index]
           
           logreg_model = LogisticRegression(penalty='l2', C=1/lambda_int[i], solver = 'lbfgs')
           logreg_model = logreg_model.fit(X_train, y_train)

           logreg_y_val_estimated = logreg_model.predict(X_val).T
           logreg_valid_errorrate[k2] = np.sum(logreg_y_val_estimated != y_val) / len(y_val)
           k2 = k2 + 1
        
        logreg_generalisation_error_rate[i] = np.sum(logreg_valid_errorrate) / len(logreg_valid_errorrate)
            
    logreg_min_error = np.min(logreg_generalisation_error_rate)
    opt_lambda_index = np.argmin(logreg_generalisation_error_rate)
    opt_lambda = lambda_int[opt_lambda_index]
    
    logreg_model = LogisticRegression(penalty='l2', C=1/lambda_int[opt_lambda_index], solver = 'lbfgs')
    logreg_model = logreg_model.fit(X_par, y_par)
    
    logreg_y_test_estimated = logreg_model.predict(X_test).T
    logistic_test_error[k1] = np.sum(logreg_y_test_estimated != y_test) / len(y_test)
    
    print('Error rate - regularized log-reg - CV1 fold {0}/{1}: {2}%'.format(k1+1, K1, np.round(100 * logistic_test_error[k1], decimals = 2)))
    print('Optimal lambda: {0}'.format(opt_lambda))
#     print()
    print('Computing CV1 fold: {0}/{1}..'.format(k1+1,K1))
    print()
    
    #Compute training and test sets for current CV fold
    X_par, y_par=X[par_index,:], y[par_index]
    X_test, y_test=X[test_index,:], y[test_index]
    
    CV2 = model_selection.KFold(n_splits=K2, shuffle = False)
    
    #Regularize Logistic regression
    lambda_int = np.power(10.,range(-5,9))
    logreg_generalisation_error_rate = np.zeros(len(lambda_int))
    
    for i in range(0, len(lambda_int)):
        k2 = 0
        logreg_valid_errorrate = np.zeros(K2)
        
        for train_index, valid_index in CV2.split(X_par):
            X_train, y_train = X_par[train_index,:], y_par[train_index]
            X_val, y_val = X_par[val_index,:], y_par[val_index]
       
            logreg_model = LogisticRegression(penalty='l2', C=1/lambda_int[i], solver = 'lbfgs')
            logreg_model = logreg_model.fit(X_train, y_train)

            logreg_y_val_estimated = logreg_model.predict(X_val).T
            logreg_valid_errorrate[k2] = np.sum(logreg_y_val_estimated != y_val) / len(y_val)
            k2 = k2 + 1
       
        logreg_generalisation_error_rate[i] = np.sum(logreg_valid_errorrate) / len(logreg_valid_errorrate)
    
    logreg_min_error = np.min(logreg_generalisation_error_rate)
    opt_lambda_index = np.argmin(logreg_generalisation_error_rate)
    opt_lambda=lambda_int[opt_lambda_index]
    
    logreg_model = LogisticRegression(penalty='l2', C=1/lambda_int[opt_lambda_index],solver = 'lbfgs')
    logreg_model = logreg_model.fit(X_par, y_par)
   
   
    logreg_y_test_estimated = logreg_model.predict(X_test).T
    logistic_test_error[k1] = np.sum(logreg_y_test_estimated != y_test) / len(y_test)
    
    print('Error rate - regularized log-reg - CV1 fold {0}/{1}: {2}%'.format(k1+1, K1, np.round(100 * logistic_test_error[k1], decimals = 2)))
    print('Optimal lambda: {0}'.format(opt_lambda))
    print()
    
    #Naïve Bayes
    
    #Baseline
    class_1_count = y_par.sum() # class 1
    class_0_count = len(y_par) - y_par.sum() # class 0
    baseline_class = float(np.argmax([class_0_count, class_1_count]))

    baseline_test_error[k1] = np.sum(y_test != baseline_class) / len(y_test)
    
    print('Error rate - baseline log-reg - CV1 fold {0}/{1}: {2}%'.format(k1+1, K1, np.round(100 * baseline_test_error[k1], decimals = 2)))
    print()
   
    k1 = k1 + 1
    print()
    print()