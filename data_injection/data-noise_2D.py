#!/bin/bash

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from mlxtend.plotting import plot_decision_regions

def opening_dataset():
    iris=load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df['target']=iris.target
    df['Flower']=df.target.apply(lambda x:iris.target_names[x]) 
    x=df.drop(['target','Flower'],axis='columns')
    clean_signal=pd.DataFrame(iris.data,columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
    mu,sigma=0,0.1
    noise=np.random.normal(mu,sigma,size=(150,4))
    noisy_signal=clean_signal+noise
    noisy_signal['target']=iris.target
    noisy_signal1=noisy_signal[noisy_signal.target==0]
    noisy_signal2=noisy_signal[noisy_signal.target==1]
    noisy_signal3=noisy_signal[noisy_signal.target==2]
    xa=x
    ya=noisy_signal.target
    xa_train,xa_test,ya_train,ya_test=train_test_split(xa,ya,test_size=0.2)
    model=SVC(C=10,kernel='poly')
    model.fit(xa_train,ya_train)
    print('Accuracy:',model.score(xa_test,ya_test)*100)
    plt.scatter(noisy_signal1['Sepal Length'],noisy_signal1['Sepal Width'],color='blue',marker='*')
    plt.scatter(noisy_signal2['Sepal Length'],noisy_signal2['Sepal Width'],color='black',marker='.')
    plt.show()

def main():
    opening_dataset()

if __name__=='__main__':
    main()
