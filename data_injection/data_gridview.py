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
    
    x=iris.data[:,:2]
    y=df.target
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
    model=SVC(C=10,kernel='poly')
    clf=model.fit(x_train,y_train)
    print('Accuracy:',model.score(x_test,y_test)*100)
    fig,ax=plt.subplots()
    title=('Decision Boundary for data without noise')
    x0,x1=x[:,0],x[:,1]
    xx,yy=meshgridd(x0,x1)

    plot_cont(ax,clf,xx,yy,cmap=plt.cm.coolwarm,alpha=0.8)
    ax.scatter(x0,x1,c=y,cmap=plt.cm.coolwarm,s=20,edgecolors="k")
    ax.set_ylabel("{}".format(feature_names[0]))
    ax.set_xlabel("{}".format(feature_names[1]))
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    plt.show()

def main():
    #meshgridd()
    #plot_cont()
    opening_dataset()

if __name__=='__main__':
    main()
