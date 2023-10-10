import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load the Iris csv data using the Pandas library
filename = 'Data/Data_for_project.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
attributeNames = df.columns[1:-2].tolist()

# Extract vector y, convert to NumPy array
raw_data = df.values  
X=raw_data[:,range(1,10)]
y=raw_data[:,10]
N=X.shape[0]
M=X.shape[1]

for i in range(0,N):
    X[i][4]= 1.0 if X[i][4]=="Present" else 0.0
    
X=X.astype(float)

    
standard=zscore(X,ddof=1)
for i in range(0,9):
    standard[:,i]/=np.std(standard[:,i],ddof=1)
    #print(np.mean(standard[:,i]),np.std(standard[:,i],ddof=1))


#classification baseline
baseline_chd1=np.sum(y)
baseline_chd0=N-baseline_chd1
baseline_class=1
if baseline_chd1<baseline_chd0:
    baseline_class=0
    
print(baseline_class)

#finding lambda for least squares regularization


#w= (standard.T @ standard) / (standard.T @ y)




from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Prepara i dati di addestramento e test (sostituisci con i tuoi dati)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizza le caratteristiche
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definisci il modello Ridge con un valore di alpha (parametro di regolarizzazione)
alpha = 1.0  # Modifica alpha a tuo piacimento
ridge_model = Ridge(alpha=alpha)

# Addestra il modello sui dati di addestramento
ridge_model.fit(X_train, y_train)

# Esegui le predizioni
y_pred = ridge_model.predict(X_test)

# Valuta il modello
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Ottieni il vettore dei pesi
weights = ridge_model.coef_
print("Vettore dei pesi:", weights)

s


