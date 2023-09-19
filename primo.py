# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:22:05 2023

@author: alberto
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt

# Load the Iris csv data using the Pandas library
filename = 'Data/Data_for_project.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
raw_data = df.values  
raw_data=raw_data[:,range(1,11)]
for i in range(0,raw_data.shape[0]):
    raw_data[i][4]= 1 if raw_data[i][4]=="Present" else 0.0

standard=raw_data.copy();

std= np.zeros(10)
for i in range(0,10):
    #print("max: "+str(raw_data[:,i].max()) + " min: "+ str(raw_data[:,i].min()))
    #plt.figure()
    #plt.hist(raw_data[:,i]);
    #plt.show()
    std[i]=np.std(raw_data[:,i],ddof=1)
    standard[:,i]=(raw_data[:,i]-np.average(raw_data[:,i]))/std[i]
    

U,S,Vh= svd(standard.astype(np.float64),full_matrices=False)

V=Vh.T

rho=(S*S)/(S*S).sum()

threshold=0.80

plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


    

    
