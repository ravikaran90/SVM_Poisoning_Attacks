#!/bin/bash

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from mlxtend.plotting import plot_decision_regions

def meshgridd(x,y,h=0.02):
    #x=iris.data[:,:2]
    #y=iris.target
    x_minimum, x_maximum=x.min()-1,x.max()+1
    y_minimum, y_maximum=y.min()-1,y.max()+1
    xx,yy=np.meshgrid(np.arange(x_minimum,x_maximum,h),np.arange(y_minimum,y_maximum,h))
    return xx, yy

def plot_cont(ax,clf,xx,yy,**params):
    z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    out=ax.contourf(xx,yy,z,**params)
    return out

def opening_dataset():
    iris=load_iris()
    feature_names=iris.feature_names[:2]
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df['target']=iris.target
    df['Flower']=df.target.apply(lambda x:iris.target_names[x]) 
    df1=df[df.target==0]
    df2=df[df.target==1]
    df3=df[df.target==2]
    
    clean_signal=pd.DataFrame(iris.data,columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
    mu,sigma=0,0.1
    noise=np.random.normal(mu,sigma,size=(150,4))
    noisy_signal=clean_signal+noise
    noisy_signal['target']=iris.target
    noisy_signal1=noisy_signal[noisy_signal.target==0]
    noisy_signal2=noisy_signal[noisy_signal.target==1]
    noisy_signal3=noisy_signal[noisy_signal.target==2]
    
    xn=iris.data[:,:2]
    ya=noisy_signal.target
    xn_train,xn_test,ya_train,ya_test=train_test_split(xn,ya,test_size=0.2)
    model=SVC(kernel='linear')
    clf=model.fit(xn_train,ya_train)
    print('Accuracy:',model.score(xn_test,ya_test)*100)
    fig,ax=plt.subplots()

    title=('Decision Boundary for data with noise')
    x0,x1=xn[:,0],xn[:,1]
    xx,yy=meshgridd(x0,x1)

    plot_cont(ax,clf,xx,yy,cmap=plt.cm.coolwarm,alpha=0.8)
    ax.scatter(x0,x1,c=ya,cmap=plt.cm.coolwarm,s=20,edgecolors="k")
    ax.set_ylabel("{}".format(feature_names[0]))
    ax.set_xlabel("{}".format(feature_names[1]))
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    plt.show()
    
def main():
    opening_dataset()

if __name__=='__main__':
    main()
