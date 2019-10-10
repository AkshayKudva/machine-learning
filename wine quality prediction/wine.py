import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
dataset=pd.read_csv('winequality-red.csv')

x=dataset.iloc[:,:11].values
y=dataset.iloc[:,11:].values

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=True,test_size=0.2)

lReg=LinearRegression()
lReg.fit(x_train,y_train)

y_pred=lReg.predict(x_test)
'''
q=0
for i in y_pred:
    y_pred[q]=round(y_pred[q],1)
    q=q+1
    '''

import statsmodels.formula.api as sm

a=np.ones((1599,1))
x=np.append(a,x,axis=1)

xopt=x[:,2:9]

#x=x[:,3:]
sm.OLS(endog=y,exog=xopt).fit().summary()

newReg=LinearRegression()

xop_Train,xop_Test,yop_Train,yop_Test=train_test_split(xopt,y,random_state=1,test_size=0.2)
newReg.fit(xop_Train,yop_Train)
newyPred=newReg.predict(xop_Test)



