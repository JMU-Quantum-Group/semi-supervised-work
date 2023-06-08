# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 17:17:13 2022

@author: zlf
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 20:27:29 2022

@author: zlf
"""

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import fu_2_qubit
from fu_2_qubit import *
data=pd.read_csv("C:/Users/WIN10/Desktop/plot_da.csv",error_bad_lines=False)

threshould=0.98
KK=data['K']
L=data['l']
U=data['u']
JJ=data['iteration']
iteration=50
start=0.3
stop=2
x=np.arange(start,stop,abs(start-stop)/iteration)

#%%

number3=3000    
X_test,y_test=Get_teststates(number3)#
X_input=pauli_express(X_test)
y=np.argmax(y_test,axis=1)
index_s=np.argwhere(y == 0)
index_e=np.argwhere(y == 1)

np.save('./Data/2-qubit/X_test'+repr(number3)+'.npy', X_test)
np.save('./Data/2-qubit/y_test'+repr(number3)+'.npy',y_test)
np.save('./Data/2-qubit/X_input'+repr(number3)+'.npy',X_input)

M=3
test_SA=Augment(X_test,M)
XX_input=pauli_express(test_SA)
np.save('./Data/2-qubit/test_SA_M3'+repr(number3)+'.npy', test_SA)
np.save('./Data/2-qubit/XX_input'+repr(number3)+'.npy',XX_input)
#%%
spervised_test=pd.DataFrame(np.zeros((10,6)),columns=["s_acc","e_acc","test_acc","ks_acc","ke_acc","ktest_acc"])
spervisedk_test=pd.DataFrame(np. zeros((10,6)),columns=["s_acc","e_acc","test_acc","ks_acc","ke_acc","ktest_acc"])
semi=pd.DataFrame(np.zeros((10,6)),columns=["s_acc","e_acc","test_acc","ks_acc","ke_acc","ktest_acc"])
for (i,K,jj,number1,number2) in zip(range(0,9),KK[0:9],JJ[0:9],L[0:9],U[0:9]):
    alpha=x[jj]
    label_size=(K+1)*number1
    def crossentropy(y_true, y_pred,L_S=label_size,lamda=alpha):#batch_size标签数据
        cc =tf.keras.losses.CategoricalCrossentropy()
        loss1 =cc(y_true[0:L_S],y_pred[0:L_S])
        loss2 =cc(y_true[L_S:],y_pred[L_S:] )
        loss=loss1+lamda*loss2
        return loss
    su_model0= keras.models.load_model('./max_model/mmodel_'+repr(number1)+'_'+repr(number2)+'.h5')
    su_model = keras.models.load_model('./max_model/pmodel_KK'+repr(K)+'+_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5')
    semi_model=keras.models.load_model('./max_model/mmodel_inter'+repr(jj)+'KK'+repr(K)+'+_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5',custom_objects={'crossentropy': crossentropy})
    
    
    test_p1=su_model0.predict(X_input,batch_size =600)
    tp1=np.argmax(test_p1,axis=1)
    s_acc1=len(np.argwhere(tp1[index_s]== 0))/number3
    e_acc1=len(np.argwhere(tp1[index_e]== 1))/number3
    spervised_test.iloc[i,0]=s_acc1
    spervised_test.iloc[i,1]=e_acc1
    spervised_test.iloc[i,2]=len(np.argwhere(tp1==y))/2/number3
    
    test_p2=su_model.predict(X_input,batch_size =600)
    tp2=np.argmax(test_p2,axis=1)
    s_acc2=len(np.argwhere(tp2[index_s]== 0))/number3
    e_acc2=len(np.argwhere(tp2[index_e]== 1))/number3
    spervisedk_test.iloc[i,0]=s_acc2
    spervisedk_test.iloc[i,1]=e_acc2
    spervisedk_test.iloc[i,2]=len(np.argwhere(tp2==y))/2/number3
    
    test_p3=semi_model.predict(X_input,batch_size =600)
    tp3=np.argmax(test_p3,axis=1)
    s_acc3=len(np.argwhere(tp3[index_s]== 0))/number3
    e_acc3=len(np.argwhere(tp3[index_e]== 1))/number3
    semi.iloc[i,0]=s_acc3
    semi.iloc[i,1]=e_acc3
    semi.iloc[i,2]=len(np.argwhere(tp3==y))/2/number3
    
    
    
    test_pk1=su_model0.predict(XX_input,batch_size = len(XX_input))
    bas1 = [test_pk1[m:m + M+1] for m in range(0, len(test_pk1),M+1)]
    p1_average=1/(M+1)*(np.sum(bas1, axis=1))
    tpk1=np.argmax(p1_average,axis=1)
    s_acck1=len(np.argwhere(tpk1[index_s]== 0))/number3
    e_acck1=len(np.argwhere(tpk1[index_e]== 1))/number3
    spervised_test.iloc[i,3]=s_acck1
    spervised_test.iloc[i,4]=e_acck1
    spervised_test.iloc[i,5]=len(np.argwhere(tpk1==y))/2/number3
    
    test_pk2=su_model.predict(XX_input,batch_size = len(XX_input))
    bas2 = [test_pk2[m:m + M+1] for m in range(0, len(test_pk2),M+1)]
    p2_average=1/(M+1)*(np.sum(bas2, axis=1))
    tpk2=np.argmax(p2_average,axis=1)
    s_acck2=len(np.argwhere(tpk2[index_s]== 0))/number3
    e_acck2=len(np.argwhere(tpk2[index_e]== 1))/number3
    spervisedk_test.iloc[i,3]=s_acck2
    spervisedk_test.iloc[i,4]=e_acck2
    spervisedk_test.iloc[i,5]=len(np.argwhere(tpk2==y))/2/number3
    
    test_pk3=semi_model.predict(XX_input,batch_size = len(XX_input))
    bas3 = [test_pk3[m:m + M+1] for m in range(0, len(test_pk3),M+1)]
    p3_average=1/(M+1)*(np.sum(bas3, axis=1))
    tpk3=np.argmax(p3_average,axis=1)
    s_acck3=len(np.argwhere(tpk3[index_s]== 0))/number3
    e_acck3=len(np.argwhere(tpk3[index_e]== 1))/number3
    semi.iloc[i,3]=s_acck3
    semi.iloc[i,4]=e_acck3
    semi.iloc[i,5]=len(np.argwhere(tpk3==y))/2/number3
semi.to_csv('./max_result/semi.csv')
spervised_test.to_csv('./max_result/spervised_test.csv')
spervisedk_test.to_csv('./max_result/spervisedk_test.csv')