# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:03:28 2022

@author: zlf
"""

import numpy as np
import qutip as Q
from qutip import *
import pandas as pd
import random 
import tensorflow as tf
import sklearn as sk #submodules does not import automatically
import sklearn.utils 
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import math
from keras.utils.np_utils import to_categorical
## 可分 纯态生成

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
    R_separable=R_separable[1:int(0.47*number)+1,:,:]
    R_entangle=R_entangle[1:int(0.53*number)+1,:,:]
    data=np.concatenate((R_separable,R_entangle), axis=0)
    Label=np.zeros((len(data),1))
    s=len(R_separable)
    Label[s:]=1 
    permutation1 = list(np.random.permutation(len(data)))
    shuffled_data1 = data[permutation1]
    shuffled_label1=Label[permutation1]
    shuffled_label1= to_categorical(shuffled_label1, num_classes=None)#转为one-hot编码
    return shuffled_data1,shuffled_label1
def Get_teststates(number1,K): #测试集生成
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
    R_separable=Augment(R_separable,K)
    R_entangle=Augment(R_entangle,K)
    return R_separable,R_entangle
def Unlabel_Gstates(number,label_da,la): #测试集生成
    unlabel_data=np.zeros((4,4)).reshape(1,4,4)
    for i in range(number2):
       M=np.random.randint(-10 ,10,size=(4,4))
       N=np.random.randint(-10,10,size=(4,4))
       H=np.dot(M+1j*N,((M+1j*N).T).conjugate()) #H      
       rho=((1/(np.trace(H)))*H).reshape(1,4,4) #密度矩阵
       unlabel_data=np.concatenate((unlabel_data,rho), axis=0)
    unlabel_data=unlabel_data[1:,:,:]
    ratio=int(2*(number*0.7-number*0.5)/len(la))
    itemindex_s = np.argwhere((la==[1,0]))#定位可分
    itemindex_s=itemindex_s[np.arange(0,len(itemindex_s),2)][:,0]
    se1 = label_da[itemindex_s]
    seundata=Augment2(se1,ratio)
    unlabel_data=np.concatenate((seundata,unlabel_data), axis=0)
    permutation1 = list(np.random.permutation(len(unlabel_data)))
    unlabel_data=unlabel_data[permutation1]
    return unlabel_data

def Augment(data,M):#对无标签数据进行3*K次增强,truth,one-hot编码
    Ub_hut=np.zeros((4,4)).reshape(1,4,4)
    for ub in data:
            ub_hut= Unitary_trans(ub,M)
            Ub_hut=np.concatenate((Ub_hut,ub_hut,ub.reshape(1,4,4)), axis=0)
    Ub_hut=Ub_hut[1:,:,:]
    return Ub_hut
def Augment2(data,M):#对无标签数据进行3*K次增强,truth,one-hot编码
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
#%%
number3=2000#测试集数据大小
test_S,test_E=Get_teststates(int(number3/2),2)
test_S=pauli_express(test_S)
test_E=pauli_express(test_E)
np.save('./Data/2-qubit/test_S.npy', test_S)
np.save('./Data/2-qubit/test_E.npy', test_E)  
truth_S=np.zeros((len(test_S),2))
truth_E=np.zeros((len(test_S),2))
truth_E[:,1]=1
truth_S[:,0]=1
vs=test_S[0:int(len(test_S)/2)]
ts=test_S[int(len(test_S)/2):]
ve=test_E[0:int(len(test_S)/2)]
te=test_E[int(len(test_S)/2):]
tvs=truth_S[0:int(len(test_S)/2)]
tts=truth_S[int(len(test_S)/2):]
tve=truth_E[0:int(len(test_S)/2)]
tte=truth_E[int(len(test_S)/2):]
        
semi_biteration=pd.DataFrame(np.zeros((30,5)),columns=["K","LN","UN","TN","semi_biteration"])

for number1 in [1000,2000,4000]:
    label_da,la=Get_states(number1)
    np.save('./Data/2-qubit/label_data_LN'+repr(number1)+'.npy', label_da)
    np.save('./Data/2-qubit/label_LN'+repr(number1)+'.npy', la)
    
    for number2 in [10000,30000,60000]:
        undata=Unlabel_Gstates(number2,label_da,la)
        np.save('./Data/2-qubit/unlabel_data_UN'+repr(number2)+'.npy', undata)
        for K in [2,4]:
            #label_data=np.load('./Data/label_data_'+repr(number1)+'.npy')
   
            label_data=Augment(label_da,K)
            #label_data= RM(label_data)
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
            np.save('./Data/2-qubit/unlabel_datape_K'+repr(K)+repr(number1)+'.npy', unlabel_datape)
            #%%参数准备
            epoch=100#监督
            Epoch=110#半监督
            rratio=[]
            iteration=40
            start=0.2
            stop=1
            x=np.arange(start,stop,abs(start-stop)/iteration)
            select_number=[]
            bestepoch=[]
            LOSS_supervised=[]
            
            acc=pd.DataFrame(np.zeros((iteration+1,6)),columns=["E_TP","teacc","test_acc","tvsacc","tveacc","val_acc"])
            pmodelloss=[]
            #%%监督
            pmodel= tf.keras.models.Sequential(
                [tf.keras.layers.Flatten(input_shape=[4,4]),  
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.Dropout(0.50),
                tf.keras.layers.Dense(512, activation='relu' ),
                tf.keras.layers.Dropout(0.50),
                tf.keras.layers.Dense(256, activation='relu' ),
                tf.keras.layers.Dropout(0.50),
                tf.keras.layers.Dense(128, activation='relu' ),
                tf.keras.layers.Dropout(0.50),
                tf.keras.layers.Dense(16, activation='relu' ),
                #tf.keras.layers.Dropout(0.50),
                tf.keras.layers.Dense(2, activation='softmax' )
                ])
            pmodel.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
            pmodel.fit(label_datape,label, epochs=epoch ,validation_split=0,shuffle=True ,batch_size = 50)
            Up=pmodel.predict(unlabel_datape,batch_size = 600)
            bas = [Up[k:k + K+1] for k in range(0, len(Up),K+1)]
            ub_average=1/(K+1)*(np.sum(bas, axis=1))
            ub=np.argmax(ub_average,axis=1)
            #ub_average=Sharpen(ub_average,T=0.5)
            index=np.argwhere(ub_average>0.95)[:,0]
            print('选择样本数：',(K+1)*len(index))
            select_number.append((K+1)*len(index))
            ratio=sum(ub[index])/len(ub[index])
            rratio.append(ratio)
            index=Index(index,K)
            ub=to_categorical(ub, num_classes=None)
            ub=QB(ub,K)
            ub=ub[index]
            tsacc=pmodel.predict(ts,batch_size =600)[1]
            teacc=pmodel.evaluate(te,tte,batch_size =600)[1]
            tvsacc=pmodel.evaluate(vs,tvs,batch_size =600)[1]
            tveacc=pmodel.evaluate(ve,tve,batch_size =600)[1]
            acc.iloc[0,0]=tsacc
            acc.iloc[0,1]=teacc
            acc.iloc[0,2]=(teacc+tsacc)/2
            acc.iloc[0,3]=tvsacc
            acc.iloc[0,4]=tveacc
            acc.iloc[0,5]=(tveacc+tvsacc)/2
            #%%迭代更新伪标签
            for jj in range(iteration): 
                iterationloss=[]
                alpha=x[jj]          #(1/((np.pi*0.4)**(1/2)))*(math.exp( (-x[jj]**2)/0.4 ))
                print('iteration=',jj)
                X=np.concatenate((label_datape,unlabel_datape[index]), axis=0)
                Y=np.concatenate((label,ub), axis=0)
                label_size=len(label_data)
                LOSS=[]
                mmodel= tf.keras.models.Sequential(
                    [tf.keras.layers.Flatten(input_shape=[4,4]),  
                    tf.keras.layers.Dense(1024, activation='relu'),
                    tf.keras.layers.Dropout(0.50),
                    tf.keras.layers.Dense(512, activation='relu' ),
                    tf.keras.layers.Dropout(0.50),
                    tf.keras.layers.Dense(256, activation='relu' ),
                    tf.keras.layers.Dropout(0.50),
                    tf.keras.layers.Dense(128, activation='relu' ),
                    tf.keras.layers.Dropout(0.50),
                    tf.keras.layers.Dense(16, activation='relu' ),
                    tf.keras.layers.Dropout(0.50),
                    tf.keras.layers.Dense(2, activation='softmax' )
                    ])
                def crossentropy(y_true, y_pred,L_S=label_size,lamda=alpha):#batch_size标签数据
                       cc =tf.keras.losses.CategoricalCrossentropy()
                       loss1 =cc(y_true[0:L_S],y_pred[0:L_S])
                       loss2 =cc(y_true[L_S:],y_pred[L_S:] )
                       loss=loss1+lamda*loss2
                       return loss
                mmodel.compile(loss=crossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
              
                mmodel.fit(X,Y, epochs=Epoch ,validation_split=0 ,batch_size = len(Y)) 
                sacc=mmodel.evaluate(test_S,truth_S,batch_size =int(number3/2))[1]
                eacc=mmodel.evaluate(test_E,truth_E,batch_size =int(number3/2))[1]
                vsacc=mmodel.evaluate(vs,tvs,batch_size =int(number3/2))[1]
                veacc=mmodel.evaluate(ve,tve,batch_size =int(number3/2))[1]
                acc.iloc[jj+1,0]=sacc
                acc.iloc[jj+1,1]=eacc
                acc.iloc[jj+1,2]=(sacc+eacc)/2
                acc.iloc[jj+1,3]=vsacc
                acc.iloc[jj+1,4]=veacc
                acc.iloc[jj+1,5]=(vsacc+veacc)/2
                Up=mmodel.predict(unlabel_datape,batch_size = len(unlabel_data))
                bas = [Up[k:k + K+1] for k in range(0, len(Up),K+1)]
                ub_average=1/(K+1)*(np.sum(bas, axis=1))
                ub=np.argmax(ub_average,axis=1)
                #ub_average=Sharpen(ub_average,T=0.5)
                index=np.argwhere(ub_average>0.95)[:,0]
                ratio=sum(ub[index])/len(ub[index])
                rratio.append(ratio)
                index=Index(index,K)
                ub=to_categorical(ub, num_classes=None)
                ub=QB(ub,K)
                ub=ub[index]
                print('选择样本数：',len(index))
                select_number.append(len(index))
            #mmodel.save('./MY_MODEL/bestmodel_K3_2000.h5')
            acc.to_csv('./Result/2-qubit/testacc_KK'+repr(K+1)+'_'+repr(number1)+'_'+repr(number2)+'.csv')#伪标签精度
            #np.save('./Result/select_number_K'+repr(K+1)+'_'+repr(number1)+'.npy', select_number)#选择加入损失的样本数
            #np.save('./Result/Ratio_K'+repr(K+1)+'_'+repr(number1)+'.npy', rratio)
            semi_biteration.iloc[0,:]=[K,number1,number2,number3,acc["val_acc"].idxmax()]
            semi_biteration.to_csv('./Result/2-qubit/semi_biteration.csv')