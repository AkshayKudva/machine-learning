import cv2
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data=load_digits()
digits=data.data
labels=data.target

plt.imshow(digits[0].reshape(8,8))
labels[34]

x_train,x_test,y_train,y_test=train_test_split(digits,labels,random_state=True,test_size=0.2)

from sklearn.linear_model import LogisticRegression

logCl=LogisticRegression()
logCl.fit(x_train,y_train)

y_pred=logCl.predict(x_test[30].reshape(1,-1))
print(y_test[30])
print(y_pred)

plt.imshow(x_test[30].reshape(8,8))



# Python script for confusion matrix creation. 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
actual = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0] 
predicted = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0] 
results = confusion_matrix(actual, predicted) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(actual, predicted)) 
print('Report : ')
print(classification_report(actual, predicted))
