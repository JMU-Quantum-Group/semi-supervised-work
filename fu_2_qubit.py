# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 18:27:21 2022

@author: zlf
"""
import numpy as np
import qutip as Q
from qutip import *
from keras.utils.np_utils import to_categorical
def Convex_Combination(data,B):#one-hot编码,b次凸组合
    #p_model=load_model('./MY_MODEL/Pmodel.h5')  
    X_hut=np.zeros((4,4)).reshape(1,4,4)
    permutation1 = list(np.random.permutation(len(data)))
    se1 = data[permutation1]
    se2=data
    for i in range(B):
        for (x1,x2) in zip(se1,se2):       
            bb=np.random.dirichlet(np.ones(2), size=1).reshape(2,1)
            x_hut=(bb[0]*x1+bb[1]*x2).reshape(1,4,4)
            X_hut=np.concatenate((X_hut,x_hut), axis=0) 
        se1=X_hut[1+i*len(se2):(i+1)*len(se2)+1,:,:]
        permutation2= list(np.random.permutation(len(se1)))
        se1=se1[permutation2]
    Sn_hut =X_hut[1:,:,:]
    return Sn_hut
def Unitary_trans(rho,K): #data,array,(,8,8),K次局部酉变换
    rho.reshape(4,4)
    UR=np.zeros((4,4)).reshape(1,4,4)
    ii=np.arange(0, 2)
    for k in range(K):
        u={}
        para=np.random.randint(-10**8, 10**8, size=(2, 3))
        for (gama,beta,delta,i) in zip(para[:,0],para[:,1],para[:,2],ii):
            u["u{0}".format(i)]=np.exp(1j*np.random.randint(-10**8, 10**8))\
                *np.dot(np.diag([np.exp(-1j*beta/2),np.exp(1j*beta/2)]),\
                        np.array([[np.cos(gama/2),-np.sin(gama/2)],[np.sin(gama/2),np.cos(gama/2)]]),\
                            np.diag([np.exp(-1j*delta/2),np.exp(1j*delta/2)]))
        Rho_augment=Q.tensor(Qobj(u['u0']),Qobj(u['u1']))*rho*\
            ((Q.tensor(Qobj(u['u0']),Qobj(u['u1']))).conj().trans()) 
        UR=np.concatenate((UR,Rho_augment.reshape(1,4,4)), axis=0)
    UR=UR[1:,:,:]
    return UR

def Get_states(number): 
    L1=0
    R_entangle=np.zeros((4,4)).reshape(1,4,4)
    R_separable=np.zeros((4,4)).reshape(1,4,4)
    y=np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
    while L1<int(number/2):
        M=np.random.randint(-10,10,size=(4,4))
        N=np.random.randint(-10,10,size=(4,4))
        H=np.dot(M+1j*N,((M+1j*N).T).conjugate()) #H      
        Rho=(1/(np.trace(H)))*H #密度矩阵
        Alpha=np.dot(np.dot(np.dot(Rho,y),Rho.conjugate()),y)#
        eigenvalue,featurevector=np.linalg.eig(Alpha)
        eigenvalue.sort()
        C_Rho=(np.max([0,(eigenvalue[3])**(1/2)-(eigenvalue[2])**(1/2)-(eigenvalue[1])**(1/2)-(eigenvalue[0])**(1/2)]))
        Rho=Rho.reshape(1,4,4)
        if C_Rho>0.0000000000001:#判断量子矩阵可分纠缠
             R_entangle=np.concatenate((R_entangle,Rho), axis=0)
        else:
             R_separable=np.concatenate((R_separable,Rho), axis=0)
             L1+=1
    R_separable=R_separable[1:int(0.50*number)+1,:,:]
    R_entangle=R_entangle[1:int(0.50*number)+1,:,:]
    data=np.concatenate((R_separable,R_entangle), axis=0)
    Label=np.zeros((len(data),1))
    s=len(R_separable)
    Label[s:]=1 
    permutation1 = list(np.random.permutation(len(data)))
    shuffled_data1 = data[permutation1]
    shuffled_label1=Label[permutation1]
    shuffled_label1= to_categorical(shuffled_label1, num_classes=None)#转为one-hot编码
    return shuffled_data1,shuffled_label1
def Get_teststates(number1): #测试集生成
    L1=0
    R_entangle=np.zeros((4,4)).reshape(1,4,4)
    R_separable=np.zeros((4,4)).reshape(1,4,4)
    y=np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
    while L1<number1:
        M=np.random.randint(-10,10,size=(4,4))
        N=np.random.randint(-10,10,size=(4,4))
        H=np.dot(M+1j*N,((M+1j*N).T).conjugate()) #H      
        Rho=(1/(np.trace(H)))*H #密度矩阵
        Alpha=np.dot(np.dot(np.dot(Rho,y),Rho.conjugate()),y)#
        eigenvalue,featurevector=np.linalg.eig(Alpha)
        eigenvalue.sort()
        C_Rho=(np.max([0,(eigenvalue[3])**(1/2)-(eigenvalue[2])**(1/2)-(eigenvalue[1])**(1/2)-(eigenvalue[0])**(1/2)]))
        Rho=Rho.reshape(1,4,4)
        if C_Rho>0.0000000000001:#判断量子矩阵可分纠缠
             R_entangle=np.concatenate((R_entangle,Rho), axis=0)
        else:
             R_separable=np.concatenate((R_separable,Rho), axis=0)
             L1+=1
    R_separable=R_separable[1:,:,:]
    R_entangle=R_entangle[1:number1+1,:,:]
    truth_S=np.zeros((number1,2))
    truth_E=np.zeros((number1,2))
    truth_S[:,0]=1
    truth_E[:,1]=1
    X_test=np.concatenate((R_separable,R_entangle), axis=0)   
    y_test=np.concatenate((truth_S,truth_E), axis=0)  
    permutation1 = list(np.random.permutation(len(X_test)))
    X_test=X_test[permutation1]
    y_test=y_test[permutation1]
    return X_test,y_test
def Unlabel_Gstates(number,label_da,la): 
    unlabel_data=np.zeros((4,4)).reshape(1,4,4)
    for i in range(number):
       M=np.random.randint(-10 ,10,size=(4,4))
       N=np.random.randint(-10,10,size=(4,4))
       H=np.dot(M+1j*N,((M+1j*N).T).conjugate()) #H      
       rho=((1/(np.trace(H)))*H).reshape(1,4,4) #密度矩阵
       unlabel_data=np.concatenate((unlabel_data,rho), axis=0)
    unlabel_data=unlabel_data[1:,:,:]
    ratio=int((number*0.7-number*0.3)/len(la))
    itemindex_s = np.argwhere((la==[1,0]))#定位可分
    itemindex_s=itemindex_s[np.arange(0,len(itemindex_s),2)][:,0]
    se1 = label_da[itemindex_s]
    seundata=Augment2(se1,ratio-1)
    CCseundata=Convex_Combination(seundata,1)
    unlabel_data=np.concatenate((seundata,CCseundata,unlabel_data), axis=0)
    permutation1 = list(np.random.permutation(len(unlabel_data)))
    unlabel_data=unlabel_data[permutation1]
    return unlabel_data[0:number]

def Augment(data,M):#
    Ub_hut=np.zeros((4,4)).reshape(1,4,4)
    for ub in data:
            ub_hut= Unitary_trans(ub,M)
            Ub_hut=np.concatenate((Ub_hut,ub_hut,ub.reshape(1,4,4)), axis=0)
    Ub_hut=Ub_hut[1:,:,:]
    return Ub_hut
def Augment2(data,M):#
    Ub_hut=np.zeros((4,4)).reshape(1,4,4)
    for ub in data:
            ub_hut= Unitary_trans(ub,M)
            Ub_hut=np.concatenate((Ub_hut,ub_hut), axis=0)
    Ub_hut=Ub_hut[1:,:,:]
    return Ub_hut
def Sharpen(qb_average,T=0.5):#qb_average为one-hot编码 
    qb=np.zeros((1,2))  
    for pj in qb_average:  
        pJ=pj**(1/T)/(pj[0]**(1/T)+(pj[1])**(1/T))
        qb=np.concatenate((qb,pJ.reshape(1,2)), axis=0)
    qb=qb[1:,:]
    return qb 
def Index(index,K):
    index1=np.zeros((1,))
    ii=np.ones((len(index1)))
    indexx1=(K+1)*index
    for k in range(K+1):
        indexk=indexx1+k*ii
        index1=np.concatenate((index1,indexk), axis=0)
    index1=np.sort(index1,axis=0)
    index1=index1.astype(np.int64)
    index1=index1[1:]
    return index1
def QB(qb,K):
    #qb = to_categorical(qb, num_classes=None)
    Yb_hut=np.zeros(((K+1)*len(qb),2)) 
    for k in range(K+1):
        Yb_hut[np.arange(k,len(Yb_hut),K+1),:]=qb
    return Yb_hut
def detect(H):#输入厄米矩阵，检测纠缠可分
    y=np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
    Alpha=np.dot(np.dot(np.dot(H,y),H.conjugate()),y)
    eigenvalue,featurevector=np.linalg.eig(Alpha)
    eigenvalue.sort()        
    e=[eigenvalue[0].real,eigenvalue[1].real,eigenvalue[2].real,eigenvalue[3].real]
    e=abs(np.array(e))
    C_Rho=np.max([0,(e[3])**(1/2)-(e[2])**(1/2)-(e[1])**(1/2)-(e[0])**(1/2)])
    #print(C_Rho)
    if C_Rho>0:#判断量子矩阵可分纠缠
         return 1 #纠缠
    else:
         return 0#可分态
def pauli_express(data): #复数矩阵转变为4*4实数矩阵
    R=np.zeros((4,4)).reshape(1,4,4)
    for i in range(len(data)):
        Rho=data[i,:,:]
        sigma=[Q.identity(2),Q.sigmax(), Q.sigmay(), Q.sigmaz()]
        a=[]
        for j in range(0,4):
           for k in range(0,4):          
               a.append(np.real(np.trace(np.dot(Rho,Q.tensor(sigma[j],sigma[k]))))) #计算
        a=np.array(a).reshape(1,4,4)
        R=np.concatenate((R,a), axis=0)
    R=R[1:,:,:]
    return R 
def RM(data):
    return np.concatenate((data.real,data.imag), axis=1)
def Gauss_data(x):
    y=np.exp(-x*x)/np.sqrt(2*np.pi)
    return y
def dataset_2qubit(number1,number2, K):
        label_da,la=Get_states(number1)
        np.save('./Data/2-qubit/label_data_'+repr(number1)+'.npy', label_da)
        np.save('./Data/2-qubit/label_'+repr(number1)+'.npy', la)
        label_da=np.load('./Data/2-qubit/label_data_'+repr(number1)+'.npy')
        la=np.load('./Data/2-qubit/label_'+repr(number1)+'.npy')
        undata=Unlabel_Gstates(number2,label_da,la)
        np.save('./Data/2-qubit/unlabel_data_'+repr(number2)+'.npy', undata)
        undata=np.load('./Data/2-qubit/unlabel_data_'+repr(number2)+'.npy')
        label_data=np.load('./Data//2-qubit/label_data_'+repr(number1)+'.npy')
        label_data=Augment(label_da,K)
        label_datape=pauli_express(label_data)
        label=QB(la,K)
        permutation1 = list(np.random.permutation(len(label_datape)))
        label_datape=label_datape[permutation1]
        label=label[permutation1]
        np.save('./Data/2-qubit/label_datape_K'+repr(K)+repr(number1)+'.npy', label_datape)
        np.save('./Data/2-qubit/labelpe_K'+repr(K)+repr(number1)+'.npy', label)
        unlabel_data=Augment(undata,K)
        #unlabel_data= RM(unlabel_data)
        unlabel_datape=pauli_express(unlabel_data)
        np.save('./Data/2-qubit/unlabel_datape_K'+repr(K)+repr(number2)+'.npy', unlabel_datape)
