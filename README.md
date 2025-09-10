# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and mathplotlib.pyplot.
2. Trace the best fit line and calculate the cost function.
3. Calculate the gradient descent and plot the graph for it.
4. Predict the profit for two population sizes
   
```
Developed by: G.Mithik jain 
RegisterNumber:212224240087
```


## Program:

```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linearREG(X1,Y,learnRate=0.01,Iteration=1000):        
    X=np.c_[np.ones(len(X1)),X1]                          
    theta=np.zeros(X.shape[1]).reshape(-1,1)              
    for _ in range(Iteration):
        predictions=(X).dot(theta).reshape(-1,1)          
        errors=(predictions-Y).reshape(-1,1)              
        theta-=learnRate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:,:-2].values) 
print(X)
X1=X.astype(float)
scaler=StandardScaler()
Y=(data.iloc[1:,-1].values).reshape(-1,1)
print(Y)
X1scaled=scaler.fit_transform(X1)
Y1scaled=scaler.fit_transform(Y)
print(X1scaled,Y1scaled)
theta=linearREG(X1scaled,Y1scaled)                             
newData=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)   
newScaled=scaler.fit_transform(newData)
prediction=np.dot(np.append(1, newScaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")


```


## Output:

## data.head()

<img width="905" height="167" alt="image" src="https://github.com/user-attachments/assets/b5d80543-b26d-4728-9b82-51578caa6d5d" />

## X values 

<img width="695" height="231" alt="image" src="https://github.com/user-attachments/assets/3844886c-d75e-412a-9572-e4edcf312cb5" /> 

## X scaled 

<img width="856" height="226" alt="image" src="https://github.com/user-attachments/assets/48eec34b-25da-42db-9588-b3aeaac21d2b" />

## Y values

<img width="315" height="276" alt="image" src="https://github.com/user-attachments/assets/e0128b4d-de9c-495a-b565-a4973ea2338e" />

## Y scaled

<img width="242" height="310" alt="image" src="https://github.com/user-attachments/assets/4a93e5ea-26f8-45ba-9671-b8861264c83e" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
