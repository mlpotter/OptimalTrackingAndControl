# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:44:10 2024

@author: siliconsynapse
"""


import scipy.io
import tensorflow as tf 
import tensorflow.compat.v1 as tfc
import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,SimpleRNN
import math 
from timeit import  time
from tqdm import tqdm
from keras import backend as ker


class RNODE():
    def __init__ (self, x0,w0,dt,Pw,Px,Qw,Qx,R,window_size,model):
        
        self.x_hat=x0
        self.w=w0
        self.dt=dt
        self.Pw=Pw
        self.Px=Px
        self.Qw=Qw
        self.Qx=Qx
        self.R=R
        self.window_size=window_size
        self.model=model
        self.nw=len(self.w)
        self.nx=len(self.nx)
        self.dfi_dw=np.zeros((self.nw))
        self.dfi_dx=np.zeros((self.nx))
        n=self.nw+self.nx
        self.F=np.zeros((n,n))
        self.F[0:self.nw,:]=np.block([[np.identity(self.nw), np.zeros((self.nw,self.nx))]])

        d=np.block([1, 0 ])
        self.H2=d
        
        output_tensor=model.output
        listOfVariableTensors=model.trainable_weights
        self.gradients_w_raw=ker.gradients(output_tensor,listOfVariableTensors)
        listOfVariableTensors=model.input
        self.gradients_x_raw=ker.gradients(output_tensor,listOfVariableTensors)
        
    def forward(self,inp):
        
        self.dx = self.model.predict(inp,verbose=0)
        sf = self.x_hat[self.nw:self.nw+2]+ self.dt*self.dx
        sf=np.array(sf)
        sf=sf.reshape((2,1))
        self.x_hat[self.nw:self.nw+2]=sf
        return 
        
    def updateGradients(self,sess):
        gradients_w = sess.run(self.gradients_w_raw, feed_dict={self.model.input: self.x_hat[self.nw:self.nw+self.ny].reshape(1,self.window_size,2)})
        gradients_x = sess.run(self.gradients_x_raw, feed_dict={self.model.input: self.x_hat[self.nw:self.nw+self.ny].reshape(1,self.window_size,2)})
        gradients_x=gradients_x[0]
            
           
            

        k=0
        for j in range(len(gradients_w)):
            weights=gradients_w[j]
            weights=np.reshape(weights,(weights.size,))
             
            self.dfi_dw[k:weights.size+k]=weights
            k=weights.size+k
     
        self.dfi_dx=np.reshape(gradients_x[0,-1,:],(int(self.nx),))
        self.dfi_dx[0]=self.dfi_dx[0]*self.dt
        self.dfi_dx[1]=self.dfi_dx[1]*self.dt +1
        a=np.block([np.zeros(self.nw),1, self.dt])
        
        b=np.block([self.dfi_dx])
        
        c=np.block([self.dt*self.dfi_dw,b])
        
        self.C_kw=np.block([[np.zeros((self.nw))],[self.dt*self.dfi_dw]])
        self.F[self.nw,:]=a
        self.F[self.nw+1,:]=c
            

        self.F3=self.F[self.nw:,0:self.nw]
        self.F4=self.F[self.nw:,self.nw:self.nw+self.nx]
        
        
        
    def predict(self,pends): 
        
        self.forward(self.x_hat[self.nw:self.nw+2].T.reshape(1,self.window_size,2))
        A=self.F4
        self.Px=np.matmul(A,np.matmul(self.Px,A.T)) + self.Qx
        
        
    def update(self,z,m):

        nm=int(self.nx/2)
        
        C=self.H2.reshape((1,self.ny))
        #C_kw=np.matmul(C,F3)
        Sx=np.matmul(C,np.matmul(self.Px,C.T)) + self.R
        Sinv=np.linalg.inv(Sx)
        Kx=np.matmul(self.Px,np.matmul(C.T,Sinv))
         
        inter=self.x_hat[-self.nx:]
        meas=np.array([])
        
      
        meas=np.append(meas,inter[[0]])
        meas=meas.reshape((nm,1))
        e1=z[m+1,:].reshape(self.nx,1)-self.x_hat[-self.nx:]
        e=z[m+1,0].reshape(nm,1)-meas
        x_=self.x_hat[-self.nx:].copy()
        self.x_hat[-self.nx:]=self.x_hat[-self.nx:]+ np.matmul(Kx,(e))
        self.Px= self.Px-np.matmul(Kx,np.matmul(C,self.Px))
       
        Sw=np.matmul(self.C_kw,np.matmul(self.Pw,self.C_kw.T)) + self.Qx
        Sinv=np.linalg.inv(Sw)
        P_int= np.matmul(self.C_kw.T,np.matmul(Sinv,self.C_kw))
        self.Pw=self.Pw-np.matmul(self.Pw,np.matmul(P_int,self.Pw))
        Kw=np.matmul(self.Pw,self.C_kw.T)
       
        e_=self.x_hat[-self.nx:]-x_
        self.x_hat[:self.nw]=self.x_hat[:self.nw]+ np.matmul(Kw,(e_))
        self.Pw= self.Pw + self.Qw

        
    
    
    
    