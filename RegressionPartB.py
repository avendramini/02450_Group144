import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn import model_selection, tree
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid)
from scipy.io import loadmat
import sklearn.linear_model as lm
from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
from scipy import stats


def rlr2_validate(X, y, lambdas, hs, cvf=10):
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
    w = np.empty((M, cvf, len(lambdas)))
    LR_train_error = np.empty((cvf, len(lambdas)))
    LR_test_error = np.empty((cvf, len(lambdas)))
    ANN_train_error = np.empty((cvf, len(hs)))
    ANN_test_error = np.empty((cvf, len(hs)))
    BL_train_error = np.empty(cvf)
    BL_test_error = np.empty(cvf)
    BL_means = np.empty(cvf)
    f = 0
    for train_index, test_index in CV.split(X, y):
        print(f"INTERNAL CROSS-VALIDATION, FOLD NUMBER: {f}")
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)
        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train

        BL_y_train_est = np.full(len(y_train), np.mean(y_train))
        BL_y_test_est = np.full(len(y_test), np.mean(y_train))

        BL_means[f]=np.mean(y_train)
        BL_train_error[f] = np.power(y_train-BL_y_train_est, 2).mean()
        BL_test_error[f] = np.power(y_test-BL_y_test_est, 2).mean()
        
        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0, 0] = 0  # remove bias regularization
            w[:, f, l] = np.linalg.solve(XtX+lambdaI, Xty).squeeze()
            # Evaluate training and test performance
            LR_train_error[f, l] = np.power(
                y_train-X_train @ w[:, f, l].T, 2).mean()
            LR_test_error[f, l] = np.power(
                y_test-X_test @ w[:, f, l].T, 2).mean()

        for h in range(0, len(hs)):
            def model(): return torch.nn.Sequential(
                torch.nn.Linear(M, hs[h]),  # M features to n_hidden_units
                torch.nn.Tanh(),   # 1st transfer function,
                torch.nn.Linear(hs[h], 1),  # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
            )
            loss_fn = torch.nn.MSELoss()
            X_train2 = torch.Tensor(X_train)
            y_train2 = torch.Tensor(y_train)
            X_test2 = torch.Tensor(X_test)
            y_test2 = torch.Tensor(y_test)
            y_train2=y_train2.unsqueeze(1)
            y_test2=y_test2.unsqueeze(1)

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train2,
                                                               y=y_train2,
                                                               n_replicates=1,
                                                               max_iter=1000)
            y_test_est = net(X_test2)
            se = (y_test_est.float()-y_test2.float())**2  # squared error
            mse = (sum(se).type(torch.float)/len(y_test2)).data.numpy()  # mean
            ANN_test_error[f, h] = mse[0]

            y_train_est = net(X_train2)
            train_se = (y_train_est.float()-y_train2.float()
                        )**2  # squared error
            train_mse = (sum(train_se).type(torch.float) /
                         len(y_train2)).data.numpy()  # mean
            # store error rate for current CV fold
            ANN_train_error[f, h] = train_mse[0]

        f = f+1

    LR_opt_val_err = np.min(np.mean(LR_test_error, axis=0))
    LR_opt_lambda = lambdas[np.argmin(np.mean(LR_test_error, axis=0))]
    LR_train_err_vs_lambda = np.mean(LR_train_error, axis=0)
    LR_test_err_vs_lambda = np.mean(LR_test_error, axis=0)
    LR_mean_w_vs_lambda = np.squeeze(np.mean(w, axis=1))

    ANN_opt_val_err = np.min(np.mean(ANN_test_error, axis=0))
    ANN_opt_h = hs[np.argmin(np.mean(ANN_test_error, axis=0))]
    ANN_train_err_vs_h = np.mean(ANN_train_error, axis=0)
    ANN_test_err_vs_h = np.mean(ANN_test_error, axis=0)

    BL_opt_val_err = np.min(BL_test_error)
    BL_opt_mean=BL_means[np.argmin(np.min(BL_test_error))]

    return LR_opt_val_err, LR_opt_lambda, LR_mean_w_vs_lambda, LR_train_err_vs_lambda, LR_test_err_vs_lambda, ANN_opt_val_err, ANN_opt_h, ANN_train_err_vs_h, ANN_test_err_vs_h,BL_opt_val_err,BL_opt_mean,BL_train_error,BL_test_error


# Load the Iris csv data using the Pandas library
filename = './Data/Data_for_project.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as
# is also described in the table in the exercise
attributeNames = df.columns[1:-2].tolist()

# Extract vector y, convert to NumPy array
raw_data = df.values

colonna_da_escludere = 9

# Seleziona tutte le colonne tranne quella da escludere
colonne_da_mantenere = [i for i in range(
    raw_data.shape[1]) if i != colonna_da_escludere]

X = raw_data[:, colonne_da_mantenere]
X[:, 0] = 1
y = raw_data[:, [colonna_da_escludere]].squeeze()
y = y.astype(float)

N = X.shape[0]
M = X.shape[1]

for i in range(0, N):
    X[i][5] = 1.0 if X[i][5] == "Present" else 0.0

X = X.astype(float)


# different values of the lambda
lambdas = np.power(10., range(-5, 9))

# REGRESSION PART B
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True)

lambdas = np.power(10., range(-5, 9))
hs=np.array([5,10,15,20])
# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
GLR_train_err = np.empty((K, 1))
GLR_test_err = np.empty((K, 1))
GBL_train_err = np.empty((K, 1))
GBL_test_err = np.empty((K, 1))
GANN_train_err = np.empty((K, 1))
GANN_test_err = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M, K))

k = 0
for train_index, test_index in CV.split(X, y):
    print(f"EXTERNAL CROSS-VALIDATION, FOLD NUMBER: {k}")
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    # CHANGE TO EVALUATE ALL 3 MODELS
    LR_opt_val_err, LR_opt_lambda, LR_mean_w_vs_lambda, LR_train_err_vs_lambda, LR_test_err_vs_lambda, ANN_opt_val_err, ANN_opt_h, ANN_train_err_vs_h, ANN_test_err_vs_h,BL_opt_val_err,BL_opt_mean,BL_train_error,BL_test_error = rlr2_validate(
        X_train, y_train, lambdas,hs, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    
    GBL_train_err[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    GBL_test_err[k] = np.square(y_test-y_train.mean()).sum(axis=0)/y_test.shape[0]
    
    lambdaI = LR_opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    GLR_train_err[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    GLR_test_err[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    
    def model(): return torch.nn.Sequential(
        torch.nn.Linear(M, ANN_opt_h),  # M features to n_hidden_units
        torch.nn.Tanh(),   # 1st transfer function,
        torch.nn.Linear(ANN_opt_h, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )
    loss_fn = torch.nn.MSELoss()
    X_train2 = torch.Tensor(X_train)
    y_train2 = torch.Tensor(y_train)
    X_test2 = torch.Tensor(X_test)
    y_test2 = torch.Tensor(y_test)
    y_train2=y_train2.unsqueeze(1)
    y_test2=y_test2.unsqueeze(1)
    

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train2,
                                                       y=y_train2,
                                                       n_replicates=1,
                                                       max_iter=1000)
    y_test_est = net(X_test2)
    se = (y_test_est.float()-y_test2.float())**2  # squared error
    mse = (sum(se).type(torch.float)/len(y_test2)).data.numpy()  # mean
    GANN_test_err[k] = mse[0]

    y_train_est = net(X_train2)
    train_se = (y_train_est.float()-y_train2.float()
                )**2  # squared error
    train_mse = (sum(train_se).type(torch.float) /
                 len(y_train2)).data.numpy()  # mean
    # store error rate for current CV fold
    GANN_train_err[k] = train_mse[0]
    
    
    k += 1
    

plt.show()
plt.title("Artificial Neural Network")
plt.xlabel("Fold")
plt.ylabel("MSE")
plt.plot(range(0,10),GANN_train_err,label="Train error")
plt.plot(range(0,10),GANN_test_err,label="Test error")
plt.legend()
plt.show()

plt.show()
plt.title("Linear Regression")
plt.xlabel("Fold")
plt.ylabel("MSE")
plt.plot(range(0,10),GLR_train_err,label="Train error")
plt.plot(range(0,10),GLR_test_err,label="Test error")
plt.legend()
plt.show()

plt.show()
plt.title("Baseline")
plt.xlabel("Fold")
plt.ylabel("MSE")
plt.plot(range(0,10),GBL_train_err,label="Train error")
plt.plot(range(0,10),GBL_test_err,label="Test error")
plt.legend()
plt.show()
