import numpy as np
import pandas as pd
from numpy import linalg as la
import argparse
import scipy.linalg  as sla
import random

from scipy.sparse import random as scirand

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

class ddstrategic_prediction:
    def __init__(self,lam=[0.0,0.0], params={'A1':[],'A2':[],'Ac1':[],'Ac2':[]},maxx=10,n=2,d=10,m=2, N_test=100, 
                 MAXITER=1000, sigma_theta=0.01,sigma_w=0.01,sigma_z1=0.01,sigma_z2=0.01,density=0.25,B=0*np.ones((10,1)),
                 seed=2,nu=1e-3, mu_w1=0, mu_w2=0, mu_theta=0):
        self.n=n
        self.d=d
        self.m=m
        self.lam1=lam[0]; 
        self.lam2=lam[1]
        self.l=[-maxx for i in range(self.d)]
        self.u=[maxx for i in range(self.d)]
        N=1000
        self.N_test=N_test
        self.MAXITER=MAXITER
        self.sigma_theta=sigma_theta
        self.sigma_w=sigma_w
        self.sigma_z1=sigma_z2
        self.sigma_z2=sigma_z1
        self.mu_w1=mu_w1
        self.mu_w2=mu_w2
        self.density=density
        self.B=B
        self.A1=params['A1']
        self.Ac1=params['Ac1']
        self.A2=params['A2']
        self.Ac2=params['Ac2']
        self.A1_hat_ = np.zeros((self.m,self.d)) #np.random.rand(m,d)
        self.Ac1_hat_ =np.zeros((self.m,self.d))# np.random.rand(m,d)
        self.A2_hat_ = np.zeros((self.m,self.d)) #np.random.rand(m,d)
        self.Ac2_hat_ = np.zeros((self.m,self.d)) #np.random.rand(m,d)
        self.seed=seed
        np.random.seed(seed)
        self.A1_hat=[self.A1_hat_]
        self.Ac1_hat=[self.Ac1_hat_]
        self.A2_hat=[self.A2_hat_]
        self.Ac2_hat=[self.Ac2_hat_]
        self.nu=nu
        self.mu_theta=mu_theta
        
    def D_theta(self):
        return np.random.normal(self.mu_theta,self.sigma_theta,size=(self.d,self.m))

    def D_w(self,player):
        if player==0:
            return np.random.normal(self.mu_w1,self.sigma_w,size=(self.m,))
        else:
            return np.random.normal(self.mu_w2,self.sigma_w,size=(self.m,))


    def proj(self,x):
        y=np.zeros(np.shape(x))
        for i in range(self.n):
            for j in range(self.d):
                if x[i][j]<=self.l[j]:
                    y[i][j]=self.l[j]
                elif self.l[j]<x[i][j] and x[i][j]<self.u[j]:
                    y[i][j]=x[i][j]
                else:
                    y[i][j]=self.u[j]
        return y

    def getgrad_so(self,x,theta):
        w=self.D_w(0)
        p1=((self.A1-theta.T).T@(theta.T@self.B.flatten()+self.A1@x[0]+self.Ac1@x[1]+w-theta.T@x[0])+self.lam1*x[0]
            +self.Ac2.T@(theta.T@B.flatten()+self.A2@x[1]+self.Ac2@x[0]+w-theta.T@x[1]))

        w=self.D_w(1)
        p2=((self.A2-theta.T).T@(theta.T@self.B.flatten()+self.A2@x[1]+self.Ac2@x[0]+w-theta.T@x[1])+self.lam2*x[1]
            +self.Ac1.T@(theta.T@self.B.flatten()+self.A1@x[0]+self.Ac1@x[1]+w-theta.T@x[0]))

        return np.vstack((p1.T,p2.T))

    def getgrad(self,x,theta):
        w=self.D_w(0)
        p1=(self.A1-theta.T).T@(theta.T@self.B.flatten()+self.A1@x[0]+self.Ac1@x[1]+w-theta.T@x[0])+self.lam1*x[0]

        w=self.D_w(1)
        p2=(self.A2-theta.T).T@(theta.T@self.B.flatten()+self.A2@x[1]+self.Ac2@x[0]+w-theta.T@x[1])+self.lam2*x[1]

        return np.vstack((p1.T,p2.T))

    def getHess(self,x,th):
        H1=(self.A1-th.T).T@(self.A1-th.T)+self.lam1*np.eye(self.d)
        H2=(self.A2-th.T).T@(self.A2-th.T)+self.lam2*np.eye(self.d)
        return H1,H2



    def D_z(self,player):
        if player==0:
            return np.random.normal(10,self.sigma_z1,size=(self.m,))
        else:
            return np.random.normal(5,self.sigma_z2,size=(self.m,))


    def update_estimate(self,x, z1_, z2_, theta, nu=1e-3, mu=1,A1hat=[],Ac1hat=[],A2hat=[],Ac2hat=[], UNCORR=False, passvals=False):
        if not(passvals):
            A1hat =self.A1_hat[-1]
            Ac1hat=self.Ac1_hat[-1]
            A2hat =self.A2_hat[-1]
            Ac2hat=self.Ac2_hat[-1]
        u1_ = np.random.normal(0,mu,size=(self.d,))
        u2_ = np.random.normal(0,mu,size=(self.d,))

        q1_ = self.D_w(0)
        q2_ = self.D_w(1)
        #print(np.shape(q1_), np.shape(q2_))
        q1 = q1_+self.A1@(x[0]+u1_)+self.Ac1@(x[1]+u2_)+theta.T@self.B.flatten()
        q2 = q2_+self.A2@(x[1]+u2_)+self.Ac2@(x[0]+u1_)+theta.T@self.B.flatten()
        z1 = z1_+self.A1@x[0]+self.Ac1@x[1]+theta.T@self.B.flatten()
        z2 = z2_+self.A2@x[1]+self.Ac2@x[0]+theta.T@self.B.flatten()

        barA1_hat = np.hstack((A1hat,Ac1hat))
        u1 = np.hstack((u1_,u2_)).reshape(self.n*self.d,1)   
        if UNCORR:
            dA1=np.diag(np.diagonal(A1hat))
            dAc1=np.diag(np.diagonal(Ac1hat))
            barA1_hat = np.hstack((dA1,dAc1))
        barA1_hat_new = barA1_hat + nu*((q1.reshape(self.m,1)-z1.reshape(self.m,1)-barA1_hat@u1)@(u1.T))

        barA2_hat = np.hstack((A2hat,Ac2hat))
        u2 = np.hstack((u2_,u1_)).reshape(self.n*self.d,1)
        if UNCORR:
            dA2=np.diag(np.diagonal(A2hat))
            dAc2=np.diag(np.diagonal(Ac2hat))
            barA2_hat = np.hstack((dA2,dAc2))
        barA2_hat_new = barA2_hat + nu*((q2.reshape(self.m,1)-z2.reshape(self.m,1)-barA2_hat@u2)@(u2.T))

        if UNCORR:
            A1_hat=np.diag(np.diagonal(barA1_hat_new[:,:self.d]))

            Ac1_hat=np.diag(np.diagonal(barA1_hat_new[:,self.d:]))
            A2_hat=np.diag(np.diagonal(barA2_hat_new[:,:self.d]))
            Ac2_hat=np.diag(np.diagonal(barA2_hat_new[:,self.d:]))
        else:
            A1_hat = barA1_hat_new[:,:self.d]
            Ac1_hat = barA1_hat_new[:,self.d:]
            A2_hat = barA2_hat_new[:,:self.d]
            Ac2_hat = barA2_hat_new[:,self.d:]
        return A1_hat, Ac1_hat, A2_hat, Ac2_hat

    def proj(self,x):
        y=np.zeros(np.shape(x))
        for i in range(self.n):
            for j in range(self.d):
                if x[i][j]<=self.l[j]:
                    y[i][j]=self.l[j]
                elif self.l[j]<x[i][j] and x[i][j]<self.u[j]:
                    y[i][j]=x[i][j]
                else:
                    y[i][j]=self.u[j]
        return y



    def getgrad_agd(self, x,theta, A1hat=[],Ac1hat=[],A2hat=[],Ac2hat=[],passvals=False ):
        if not(passvals):
            A1hat=self.A1_hat[-1]
            Ac1hat=self.Ac1_hat[-1]
            A2hat=self.A2_hat[-1]
            Ac2hat=self.Ac2_hat[-1]
        w=self.D_w(0)
        p1=(A1hat-theta.T).T@(theta.T@self.B.flatten()+A1hat@x[0]+Ac1hat@x[1]+w-theta.T@x[0])+self.lam1*x[0]

        w=self.D_w(1)
        p2=(A2hat-theta.T).T@(theta.T@self.B.flatten()+A2hat@x[1]+Ac2hat@x[0]+w-theta.T@x[1])+self.lam2*x[1]

        return np.vstack((p1.T,p2.T))

    def get_loss(self,x,z1_,z2_,theta):
        z1 = z1_+self.A1@(x[0])+self.Ac1@(x[1])+theta.T@self.B.flatten()
        l1 = 0.5*z1.T@z1+self.lam1*la.norm(x[0])
        z2 = z2_+self.A2@(x[1])+self.Ac2@(x[0])+theta.T@self.B.flatten()
        l2 = 0.5*z2.T@z2 +self.lam2*la.norm(x[1])
        return l1,l2

    def getgrad_rgd(self,x,z1_,z2_,theta):
        p1=-theta@(z1_-theta.T@x[0])+self.lam1*x[0]
        p2=-theta@(z2_-theta.T@x[1])+self.lam2*x[1]
        return  np.vstack((p1.T,p2.T))