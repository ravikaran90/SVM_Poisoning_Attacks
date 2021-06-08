#!/bin/bash

import sys
import numpy as np

def hinge_loss_matrix(x,actual_type):
    KM=np.zeros(shape=(x.shape[0],x.shape[0]))
    if actual_type=='linear':
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                KM[i,j]=np.dot(x[i,],x[j,])
    else:
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                KM[i,j]=np.dot(x[i,],x[j,])*np.dot(x[i,]*x[j,])
    return KM

def main():
    total_data=np.genfromtxt(sys.argv[1],delimiter=',')  
    n=total_data.shape[0]
    X=np.ones(shape=(n,3))
    
    K=hinge_loss_matrix(X,sys.argv[2])
    print('SVM using gradient ascent with %s kernel'%sys.argv[2])
    
    #Step Size
    step_size=1/np.diag(K)
    iter=0
    
    #Initialize Alphas
    alpha=np.zeros(n)
    
    #Difference
    diff=1
    eps=0.0001
    C=10
    
    while(diff>eps):
        alpha0=alpha.copy()
        for k in range(n):
            alpha[k]=alpha[k]+step_size[k]*(1-total_data[k,2]*sum(alpha*total_data[:,2]*K[:,k]))
            if alpha[k]<0:
                alpha[k]=0
            else:
                alpha[k]>C;
                alpha[k]=C
        iter=iter+1
        diff=sum((alpha-alpha0)*(alpha-alpha0))
        print(iter,diff)
        
    print('Support Vectors:')
    for k in range(n):
        if alpha[k]!=0:
            print('Sample %d: %.1f, %.1f;class:%d,a:%f\n'%(k+1,total_data[k,0],total_data[k,1],total_data[k,2],alpha[k]))

    print('Total Number of Support Vectors:%d\n'%sum(alpha!=0))
 

if __name__=='__main__':
    main()
