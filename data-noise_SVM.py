#/bin/bash

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def func():
    iris=load_iris()
    #After Data Modification
    datafr=pd.read_csv('iris_dataset.csv')
    datafr1=datafr[datafr.target==0]
    datafr2=datafr[datafr.target==1]
    datafr3=datafr[datafr.target==2]
    
    y_new=datafr.target
    x_new=datafr.drop(['target','Flower'],axis='columns')
    
    x_new_train,x_new_test,y_new_train,y_new_test=train_test_split(x_new,y_new,test_size=0.2)
    model=SVC()
    model.fit(x_new_train,y_new_train)
    print('Accuracy:',model.score(x_new_test,y_new_test)*100)

    plt.scatter(datafr1['sepal length (cm)'],datafr1['sepal width (cm)'],color='black',marker='.')
    plt.scatter(datafr2['sepal length (cm)'],datafr2['sepal width (cm)'],color='darkorange',marker='o')
    plt.show()

def main():
    func()

if __name__=='__main__':
    main()
