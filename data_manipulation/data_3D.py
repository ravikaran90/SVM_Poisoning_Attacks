import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

def plotting_3D(X,Y):
    X=X[np.logical_or(Y==0,Y==1)]
    Y=Y[np.logical_or(Y==0,Y==1)]
    
    model=SVC(kernel='linear')
    clf=model.fit(X,Y)

    z=lambda x,y:(-clf.intercept_[0]-clf.coef_[0][0]*x-clf.coef_[0][1]*y)/clf.coef_[0][2]
    tmp=np.linspace(-5,5,30)
    x,y=np.meshgrid(tmp,tmp)

    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot3D(X[Y==0,0],X[Y==0,1],X[Y==0,2],'ob')
    ax.plot3D(X[Y==1,0],X[Y==1,1],X[Y==1,2],'sr')
    ax.plot_surface(x,y,z(x,y))
    ax.view_init(30,60)
    plt.show()

def opening_dataset():
    iris=load_iris()
    X=iris.data[:,:3]
    Y=iris.target
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

    plotting_3D(X,Y)

def main():
    opening_dataset()

if __name__=='__main__':
    main()
