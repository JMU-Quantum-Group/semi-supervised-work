# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:10:22 2022

@author: zlf
"""
import numpy as np
import random 
import sys
sys.path.append("C:\\Users\\好梦难追\\Desktop\\semi-supervised entanglement work")
import Generic_functions_ghzn
from Generic_functions_ghzn import *
from keras.utils.np_utils import to_categorical
class generate():
    def __init__(self,number1,number2,n):
        self.I=np.eye(2**n)
        self.number1,self.number2=number1,number2
        self.n=n
        self.ghz_n=((1/np.sqrt(2))*(self.I[0,:]+self.I[2**self.n-1,:])).reshape(2**self.n,1) #|GHZ_n>
        self.N=np.zeros(self.n+1) #N is a list in which the elements are k-separable boundary values, p ≤ N[i] 
        self.N[0]=1
        self.GHZ={}
        self.a=7/8
        for i in range(1,self.n):  
            if i+1==2:
                self.N[i]=(2**(self.n-1)-1)/(2**self.n-1)
            elif i+1>=(self.n+1)/2:
                self.N[i]=1/(1+(2*(i+1)-self.n)*(2**(self.n-1))/self.n)
            elif i+1==self.n:
                self.N[i]=1/(1+2**(self.n-1))
            else:
                self.N[i]=9/41
        self.N=list(reversed(self.N))
        print(self.N)
        self.step=0.0005
    def Ghz(self): 
        N=self.N
        la,ratio={},{}
        for i in range(int((self.number1/2))):
            for j in range(self.n-3,self.n-1):
                if j==(self.n-3):
                    g=random.uniform(N[j],N[j]+(self.a*(N[j+2]-N[j])/4))
                else:
                    g=random.uniform(N[j+1]-(self.a*2*(N[j+1]-N[j-1])/3),N[j+1])     
             
                ghz=((1-g)*self.I/(2**self.n)+g*(np.dot(self.ghz_n,(self.ghz_n.conj().T)))).reshape(1,2**self.n,2**self.n)
                if i==0:
                    self.GHZ["{0}".format(j-(self.n-3)+1)]=ghz
                else:
                    self.GHZ["{0}".format(j-(self.n-3)+1)]=np.concatenate((self.GHZ["{0}".format(j-(self.n-3)+1)],ghz), axis=0)
        for j in range(2):
            la["{0}".format(j+1)]=np.zeros((len(self.GHZ["{0}".format(j+1)]),2))
            la["{0}".format(j+1)][:,-(1+j)]=1
            if j!=1:
                ratio["{0}".format(j+1)]=int(round(((1-(N[j+1]-N[j])/N[-2])*(1-N[-2])*self.number2)/len(self.GHZ["{0}".format(j+1)]),0))
        return self.GHZ,la,ratio  #return 
    def Ghzt(self): 
        N=self.N
        la={}
        
        for i in range(int((self.number1/2))):
            for j in range(self.n-3,self.n-1):
                if j==(self.n-3):
                    g=random.uniform(N[j],N[j]+(self.a*(N[j+2]-N[j])/4)+(self.a/10*(N[j+2]-N[j])/4))
                else:
                    g=random.uniform(N[j+1]-(self.a*2*(N[j+1]-N[j-1])/3)-(self.a/20*2*(N[j+1]-N[j-1])/3),N[j+1])
                ghz=((1-g)*self.I/(2**self.n)+g*(np.dot(self.ghz_n,(self.ghz_n.conj().T)))).reshape(1,2**self.n,2**self.n)
                if i==0:
                    self.GHZ["{0}".format(j-(self.n-3)+1)]=ghz
                else:
                    self.GHZ["{0}".format(j-(self.n-3)+1)]=np.concatenate((self.GHZ["{0}".format(j-(self.n-3)+1)],ghz), axis=0)
        for j in range(2):
            la["{0}".format(j+1)]=np.zeros((len(self.GHZ["{0}".format(j+1)]),2))
            la["{0}".format(j+1)][:,-(1+j)]=1
            
        return self.GHZ,la #return 
    def bound_states(self): 
        G=np.arange(self.N[self.n-3],self.N[self.n-1],self.step)
        for (i,g) in zip(range(len(G)),G):
            ghz=((1-g)*self.I/(2**self.n)+g*(np.dot(self.ghz_n,(self.ghz_n.conj().T)))).reshape(1,2**self.n,2**self.n)
            if i==0:
               bound_states=ghz
            else:
               bound_states=np.concatenate((bound_states,ghz), axis=0)
        return  bound_states
    def calculate_boundary(self,bpk): # input prediction of GHZ states
        bpk2=to_categorical(bpk, num_classes=2)
        i,j,bound_value=0,1,[]
        while j!=(3):
            A=np.argwhere(bpk2[:,-j]==1)
            b_sum10=20
            while b_sum10>10 and (i<len(A)-6):
                b_sum10=A[i+6]-A[i]
                if b_sum10>10:
                    i+=1
            j+=1
            #print(A[i])
            if len(A)==0:
                bound_value.append(None)
            else:
                bound_value.append(self.step*A[i]+self.N[self.n-3])
        return bound_value
#a,b,c=generate(number1=100,number2=200,n=3).Ghz()
class get_data(generate):
    def __init__(self,number1,number2,n,K):   
        super(get_data,self).__init__(number1,number2,n)
        self.GHZ,self.la,self.ratio=generate(number1=self.number1,number2=self.number2,n=self.n).Ghz()
        self.K=K
    def data_permutation(self):
        for j in range(2):
            if j==0:
                label,train_ghz=self.la["{0}".format(j+1)],self.GHZ["{0}".format(j+1)]
            else:
                label=np.concatenate((label,self.la["{0}".format(j+1)]), axis=0)
                train_ghz=np.concatenate((train_ghz,self.GHZ["{0}".format(j+1)]), axis=0)
        aug=Augmentation_Strategies(data=train_ghz,K=self.K,n=self.n)
        train_ghz=aug.Augment(A=1)
        label=aug.QB(label,False,d=2)
        permutation= list(np.random.permutation(len(train_ghz)))
        train_ghz=train_ghz[permutation]
        label=label[permutation]
        return train_ghz,label
    def Un_State(self):
        sea={}
        for i in range(int(self.number2)):
            g1=random.uniform(0,1)
            ghz1=((1-g1)*self.I/(2**self.n)+g1*(np.dot(self.ghz_n,(self.ghz_n.conj().T)))).reshape(1,2**self.n,2**self.n)
            if i==0:
                UnS=ghz1
            else:
                UnS=np.concatenate((UnS,ghz1), axis=0)
        for j in range(1):
            sea["{0}".format(j+1)]=Augmentation_Strategies(data=self.GHZ["{0}".format(j+1)],n=self.n).Convex_Combination(B=self.ratio["{0}".format(j+1)])
            UnD=sea["{0}".format(j+1)]
        UnD=np.concatenate((UnD,UnS), axis=0)
        permutation=list(np.random.permutation(len(UnD)))
        UnD=UnD[permutation]
        UnD=Augmentation_Strategies(data=UnD[0:self.number2],K=self.K,n=self.n).Augment(A=1)
        return UnD
    def test_Ghz(self,number3):
        GHZ_test,test_label=generate(number1=number3,number2=0,n=self.n).Ghzt()
        test_ghz={}
        for j in range(2):
                test_ghz["{0}".format(j+1)]=Augmentation_Strategies(data=GHZ_test["{0}".format(j+1)],K=1,n=self.n).Augment(A=1)
        return test_ghz,test_label
#test_ghz,test_label=get_data(number1=100,number2=200,n=3,K=2). test_Ghz(3000)
def col_ca(n):
    col=[]
    C=list(reversed(range(2,n+1)))
    col.append("test_eval")
    col.append("test_AV")
    for i in C:
        col.append("test_{0}sepa".format(i))
    col.append("test_gme")
    return col   
def col_ca2(n):
    col=[]
    C=list(reversed(range(2,n+1)))
    for i in C:
        col.append("bound_bk{0}".format(i))
    return col     
   
    
    
    
    
    
    
    
    
    