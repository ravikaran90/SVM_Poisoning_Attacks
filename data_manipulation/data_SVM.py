#/bin/bash

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def func():
    iris=load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df['target']=iris.target
    df1=df[df.target==0]
    df2=df[df.target==1]
    df3=df[df.target==2]
    x=df.drop(['target'],axis='columns')
    y=df.target
    #print(y)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    model=SVC()
    model.fit(x_train,y_train)
    print('Accuracy:',model.score(x_test,y_test)*100)

    plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='brown',marker='.')
    plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='orange',marker='o')
    plt.show()

def main():
    func()

if __name__=='__main__':
    main()
