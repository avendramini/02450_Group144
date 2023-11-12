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
from scipy import stats as st
from sklearn import model_selection, tree
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import torch
from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net
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

#Initializing confidence intervals
CI_BLvsLogistic = np.empty(K1, dtype=object)

#BL VS ANN
CI_BLvsANN = np.empty(K1, dtype=object)

#ANN VS LR
CI_ANNvszLogistic  = np.empty(K1, dtype=object)

#Initialize p-values
p_BLvsLogistic= np.empty((K1,1))
#BL VS ANN
p_BLvsANN = np.empty((K1,1))
#ANN VS LR
p_ANNvszLogistic= np.empty((K1,1))

#Add the column of ones
X = np.hstack((ones, z_score))

#Initialize variable
logistic_test_error = np.zeros(K1)
nbayes_test_error = np.zeros(K1)
baseline_test_error = np.zeros(K1)

def rlr2_validate(X, y, hs, cvf=10): #Modify to include lambda for logistic regression
    ''' Validate regularized linear regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation error) from 'lambdas' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all lambdas, MSE train&validation errors for all lambdas.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.

        Parameters:
        X       training data set
        y       vector of values
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds     

        Returns:
        LR_opt_val_err         validation error for optimum lambda
        LR_opt_lambda          value of optimal lambda
        LR_mean_w_vs_lambda    weights as function of lambda (matrix)
        LR_train_err_vs_lambda train error as function of lambda (vector)
        LR_test_err_vs_lambda  test error as function of lambda (vector)
    '''
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    ANN_train_error = np.empty((cvf, len(hs)))
    ANN_test_error = np.empty((cvf, len(hs)))
    
    f = 0
    errors = []
    for train_index, test_index in CV.split(X, y):
        #print(f"INTERNAL CROSS-VALIDATION, FOLD NUMBER: {f}")
        X_train = torch.Tensor(X[train_index,:] )
        y_train = torch.Tensor(y[train_index] )
        X_test = torch.Tensor(X[test_index,:] )
        y_test = torch.Tensor(y[test_index] )

        # Standardize the training and set set based on training set moments
        mu = torch.mean(X_train[:, 1:], 0)
        sigma = torch.std(X_train[:, 1:], 0)
        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
       
        
        

        for h in range(0, len(hs)):
            print("Number of hidden layers:" + str(hs[h]))
            def model(): return torch.nn.Sequential(
                torch.nn.Linear(M, hs[h]),  # M features to n_hidden_units
                torch.nn.Tanh(),   # 1st transfer function,
                torch.nn.Linear(hs[h], 1),  # n_hidden_units to 1 output neuron
                torch.nn.Sigmoid()# final tranfer function
                # no final tranfer function, i.e. "linear output"
            )
            loss_fn = torch.nn.MSELoss()
            X_train2 = torch.Tensor(X_train)
            y_train2 = torch.Tensor(y_train)
            X_test2 = torch.Tensor(X_test)
            y_test2 = torch.Tensor(y_test)
            y_train2=y_train2.unsqueeze(1)
            y_test2=y_test2.unsqueeze(1)
            
            #Trying all hs
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train2,
                                                               y=y_train2,
                                                               n_replicates=1,
                                                               max_iter=3000)
            print('\n\tBest loss: {}\n'.format(final_loss))
            
            y_sigmoid = net(X_test) # activation of final note, i.e. prediction of network
            y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
            y_test = y_test.type(dtype=torch.uint8)
            # Determine  errors and error rate
            e = (y_test_est != y_test)
            ANN_error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
            errors.append(ANN_error_rate) # store error rate for current CV fold 
            
            

        f = f+1

    

    ANN_opt_val_err = np.min(np.mean(ANN_test_error, axis=0))
    ANN_opt_h = hs[np.argmin(np.mean(ANN_test_error, axis=0))]
    ANN_train_err_vs_h = np.mean(ANN_train_error, axis=0)
    ANN_test_err_vs_h = np.mean(ANN_test_error, axis=0)
    print("Optimal validation error" + str(ANN_opt_val_err))
    print("Optimal number of hidden layers:" + str(ANN_opt_h))
    print("Train error vs hidden layer:" + str(ANN_train_err_vs_h))
    print("Test error vs hidden layer:" + str(ANN_test_err_vs_h))
    return e, ANN_opt_val_err, ANN_opt_h, ANN_train_err_vs_h, ANN_test_err_vs_h



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
    print()
   
    
    
    
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
    
    #ANN
    
    e, ANN_opt_val_err, ANN_opt_h, ANN_train_err_vs_h, ANN_test_err_vs_h = rlr2_validate(X_train, y_train, range(1,2), cvf=10)
    
    e = np.array(e)
    ANN_opt_val_err = np.array(ANN_opt_val_err)
    ANN_opt_h = np.array(ANN_opt_h)
    ANN_train_err_vs_h = np.array(ANN_train_err_vs_h) 
    ANN_test_err_vs_h = np.array(ANN_test_err_vs_h)
    
    #COMPARISON
    zBL=np.abs(y_test - baseline_class  ) ** 2
    zLogistic=np.abs(y_test-logreg_y_test_estimated)**2
    zANN=(e)**2 #Needs to be 47x1
    
    alpha = 0.05
    CIBL = st.t.interval(1-alpha, df=len(zBL)-1, loc=np.mean(zBL), scale=st.sem(zBL))# Confidence interval    
    CILogistic = st.t.interval(1-alpha, df=len(zLogistic)-1, loc=np.mean(zLogistic), scale=st.sem(zLogistic))
    CIANN = st.t.interval(1-alpha, df=len(zANN)-1, loc=np.mean(zANN), scale=st.sem(zANN))
    
    k=0
    #BL VS LR
    z = zBL - zLogistic
    CI_BLvsLogistic[k] = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p_BLvsLogistic[k] = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
    
    #BL VS ANN
    z = zBL - zANN
    CI_BLvsANN[k] = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p_BLvsANN[k] = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
    
    #ANN VS LR
    z = zANN - zLogistic
    CI_ANNvszLogistic[k] = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p_ANNvszLogistic[k] = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
    
    