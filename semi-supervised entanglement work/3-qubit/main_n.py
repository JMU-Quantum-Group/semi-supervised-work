# -*- coding: utf-8 -*-
"""
Created on Sat May 14 21:00:21 2022

@author: zlf
"""
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.utils.np_utils import to_categorical
import function_n_qubit
from function_n_qubit import *
import sys
sys.path.append("C:\\Users\\WIN10\\Desktop\\semi-supervised entanglement work")#add path of Generic_functions_ghzn
import Generic_functions_ghzn
from Generic_functions_ghzn import *
from tensorflow import keras
n=6 #n-qubit 
number3=500*n
K=5
number1=200
number2=1000
data_gen=get_data(number1=number1,number2=number2,n=n,K=K)
test_ghz,test_label=data_gen.test_Ghz(number3)
unlabel_data=data_gen.Un_State()
la_data,lA=data_gen.data_permutation()
bound_state=data_gen.bound_states()
bound_state=Augmentation_Strategies(data=bound_state,K=1,n=n).Augment(A=1)

la_data1,unlabel_data1,test_ghz1,bound_state1=Feature_Trans().RM(la_data,unlabel_data,test_ghz,bound_state,n=n)
epoch=120#监督
iteration=30
acc=pd.DataFrame(np.zeros((iteration+2,n+2)),columns=[list(col_ca(n))])

bound_bk=pd.DataFrame(np.zeros((iteration+2,n-1)),columns=[list(col_ca2(n))])
#%%
ppmodel= tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=[2**n,2**n]),  
    tf.keras.layers.Dense(512, activation='relu' ),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(256, activation='relu' ),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(128, activation='relu' ),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(16, activation='relu' ),
    tf.keras.layers.Dense(n, activation='softmax' )
    ])
ppmodel.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
ppmodel.fit(la_data,lA, epochs=epoch ,validation_split=0,shuffle=True ,batch_size = 2000)
a1=[]
for i in range(n):
    a1.append(ppmodel.evaluate(test_ghz["{0}".format(i+1)],test_label["{0}".format(i+1)])[1])
print(np.mean(a1))  
bppk=ppmodel.predict_classes(bound_state,batch_size = 2000)
bound_vpp=generate(number1=number1,number2=number2,n=n).calculate_boundary(bppk)
print(bound_vpp)
bound_bk.iloc[0]=bound_vpp[1:]

acc.loc[0,"test_gme"]=ppmodel.evaluate(test_ghz["{0}".format(n)],test_label["{0}".format(n)])[1]
C=list(reversed(range(2,n+1)))
for i,j in zip(range(n-1),C):
    acc.loc[0,"test_{0}sepa".format(j)]=ppmodel.evaluate(test_ghz["{0}".format(i+1)],test_label["{0}".format(i+1)])[1]
    acc.loc[0].at["test_AV"]=acc.loc[0].at["test_AV"]+acc.loc[0].at["test_{0}sepa".format(j)]
acc.loc[0,"test_eval"]= (acc.loc[0].at["test_3sepa"]+acc.loc[0].at["test_2sepa"])/2
acc.loc[0].at["test_AV"]=(acc.loc[0].at["test_AV"]+acc.loc[0].at["test_gme"])/n
#%%
Epoch=160#半监
threshould=0.95
rratio=[]
start=-1
stop=1
x=np.arange(start,stop,abs(start-stop)/iteration)
y=Gauss_fun(x)
select_number=[]
augment=Augmentation_Strategies(K=K,n=n)
pmodel= tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=[la_data1.shape[1]]), 
     tf.keras.layers.Dense(200, activation='relu' ),
     tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Dense(100, activation='relu' ),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Dense(128, activation='relu' ),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Dense(16, activation='relu' ),
    tf.keras.layers.Dense(n, activation='softmax' )
    ])
pmodel.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
hostoryp=pmodel.fit(la_data1,lA, epochs=epoch ,validation_split=0,shuffle=True ,batch_size = 2000)
#print(hostoryp.history['val_accuracy'][epoch-1])
Up=pmodel.predict(unlabel_data1,batch_size = 2000)
bas = [Up[k:k + K] for k in range(0, len(Up),K)]
ub_average=1/(K)*(np.sum(bas, axis=1))
ub=np.argmax(ub_average,axis=1)
#ub_average=Sharpen(ub_average,T=0.5)
index=np.argwhere(ub_average>threshould)[:,0]
print('选择样本数：',(K)*len(index))
select_number.append((K)*len(index))
index=augment.Index(index)
ub=to_categorical(ub, num_classes=n)
ub=augment.QB(ub,T=False)
ub=ub[index]

bpk=pmodel.predict_classes(bound_state1,batch_size = 2000)
bound_vp=generate(number1=number1,number2=number2,n=n).calculate_boundary(bpk)
bound_bk.iloc[1]=bound_vp[1:]
acc.loc[1,"test_gme"]=pmodel.evaluate(test_ghz1["{0}".format(n)],test_label["{0}".format(n)])[1]
for i,j in zip(range(n-1),C):
    acc.loc[1,"test_{0}sepa".format(j)]=pmodel.evaluate(test_ghz1["{0}".format(i+1)],test_label["{0}".format(i+1)])[1]
    acc.loc[1].at["test_AV"]=acc.loc[1].at["test_AV"]+acc.loc[1].at["test_{0}sepa".format(j)]
acc.loc[1,"test_eval"]= (acc.loc[1].at["test_3sepa"]+acc.loc[1].at["test_2sepa"])/2
#acc.loc[0,"bound_3k"]=bound_vp
acc.loc[1].at["test_AV"]=(acc.loc[1].at["test_AV"]+acc.loc[1].at["test_gme"])/n
pmodel.save('./model_n_qubit/'+repr(n)+'_qubit'+'_pmodel_'+'K'+repr(K)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5')
for jj in range(iteration): 
    alpha=y[jj]          #(1/((np.pi*0.4)**(1/2)))*(math.exp( (-x[jj]**2)/0.4 ))
    print('iteration=',jj)
    X=np.concatenate((la_data1,unlabel_data1[index]), axis=0)
    Y=np.concatenate((lA,ub), axis=0)
    label_size=len(la_data1)
    mmodel= tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=[la_data1.shape[1]]),
     tf.keras.layers.Dense(200, activation='relu' ),
     tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Dense(100, activation='relu' ),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Dense(128, activation='relu' ),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Dense(16, activation='relu' ),
    tf.keras.layers.Dense(n, activation='softmax' )
    ])
    def mycrossentropy(y_true, y_pred,L_S=label_size,lamda=alpha):#batch_size标签数据
           cc =tf.keras.losses.CategoricalCrossentropy()
           loss1 =cc(y_true[0:L_S],y_pred[0:L_S])
           loss2 =cc(y_true[L_S:],y_pred[L_S:] )
           loss=loss1+lamda*loss2
           return loss
    mmodel.compile(loss=mycrossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
    historym=mmodel.fit(X,Y,epochs=Epoch ,validation_split=0,batch_size = len(Y))    
    Up=mmodel.predict(unlabel_data1,batch_size = len(unlabel_data1))
    bas = [Up[k:k + K] for k in range(0, len(Up),K)]
    ub_average=1/(K)*(np.sum(bas, axis=1))
    ub=np.argmax(ub_average,axis=1)
    #ub_average=Sharpen(ub_average,T=0.5)
    index=np.argwhere(ub_average>threshould)[:,0]
    print('选择样本数：',(K)*len(index))
    select_number.append((K)*len(index))
    index=augment.Index(index)
    ub=to_categorical(ub, num_classes=n)
    ub=augment.QB(ub,T=False)
    ub=ub[index]
    acc.loc[jj+2,"test_gme"]=mmodel.evaluate(test_ghz1["{0}".format(n)],test_label["{0}".format(n)])[1]
    bmk=mmodel.predict_classes(bound_state1,batch_size = len(bound_state1))
    bound_vm=generate(number1=number1,number2=number2,n=n).calculate_boundary(bmk)
    bound_bk.iloc[jj+2]=bound_vm[1:]
    #acc.loc[jj+1,"bound_3k"]=bound_vm
    for i,j in zip(range(n-1),C):
        acc.loc[jj+2,"test_{0}sepa".format(j)]=mmodel.evaluate(test_ghz1["{0}".format(i+1)],test_label["{0}".format(i+1)])[1]
        acc.loc[jj+2].at["test_AV"]=acc.loc[jj+2].at["test_AV"]+acc.loc[jj+2].at["test_{0}sepa".format(j)]
    acc.loc[jj+2].at["test_AV"]=(acc.loc[jj+2].at["test_AV"]+acc.loc[jj+2].at["test_gme"])/n
    acc.loc[jj+2,"test_eval"]=( acc.loc[jj+2].at["test_3sepa"]+acc.loc[jj+2].at["test_2sepa"])/2
    mmodel.save('./model_n_qubit/'+repr(n)+'_qubit'+'mmodel_'+'KK'+repr(K)+'_'+repr(jj)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5')
    if jj>=1 and (acc.loc[jj].at["test_eval"]>acc.loc[jj-1].at["test_eval"]):
        mmodel.save('./max_model/'+repr(n)+'_qubit'+'best_KK'+repr(K)+'_'+repr(jj)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5') 
    acc.to_csv('./Result/'+repr(n)+'_qubit'+'_testacc_kK'+repr(K)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.csv')
    bound_bk.to_csv('./Result/'+repr(n)+'_qubit'+'_bound_bk_kK'+repr(K)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.csv')
#%%
