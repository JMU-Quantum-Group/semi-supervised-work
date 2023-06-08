# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:22:35 2022

@author: zlf
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 14 21:00:21 2022

@author: zlf
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from keras.utils.np_utils import to_categorical
import fu_3_qubit
from fu_3_qubit import *
## 可分 纯态生成

'''
def test_data(number):
    Ghz1=np.zeros((8,8)).reshape(1,8,8)
    Ghz2=np.zeros((8,8)).reshape(1,8,8)
    Ghz3=np.zeros((8,8)).reshape(1,8,8)
    ghz_3=((1/np.sqrt(2))*(I[0,:]+I[7,:])).reshape(8,1)
    G=np.linspace(0.000001,1,number)
    G1=G[np.where(G<0.2)]
    G2=G[np.where((G>=0.2) &(G<0.429))]
    G3=G[np.where(G>=0.429)]
    for g1 in G1:
        ghz1=(1-g1)*I/8+g1*(np.dot(ghz_3,(ghz_3.conj().T)))
        Ghz1=np.concatenate((Ghz1,ghz1.reshape(1,8,8)), axis=0)
    for g2 in G2:
         ghz2=(1-g2)*I/8+g2*(np.dot(ghz_3,(ghz_3.conj().T)))
         Ghz2=np.concatenate((Ghz2,ghz2.reshape(1,8,8)), axis=0)
    for g3 in G3:    
        ghz3=(1-g3)*I/8+g3*(np.dot(ghz_3,(ghz_3.conj().T)))
        Ghz3=np.concatenate((Ghz3,ghz3.reshape(1,8,8)), axis=0)
    Ghz1=Ghz1[1:]
    Ghz2=Ghz2[1:]
    Ghz3=Ghz3[1:]
    la1=np.zeros((len(Ghz1),3))
    la1[:,0]=1
    la2=np.zeros((len(Ghz2),3))
    la2[:,1]=1
    la3=np.zeros((len(Ghz3),3))
    la3[:,2]=1
    return Ghz1,Ghz2,Ghz3,la1,la2,la3,G
'''

#%%
t=0
number3=3000
K=10
K2=2
K0=10
number1=200
number2=2000
M=[3000]
label_data,label=Ghz2(number1,0)
la_data=Augment2(label_data, K0)
lA=QB(label,K0)
ghz_val1,la_val1,ghz_val2,la_val2,ghz_val3,la_val3=Ghz2(number3,1)
ghz_val1,la_val1,ghz_val2,la_val2,ghz_val3,la_val3=Augment2(ghz_val1,K2),QB(la_val1,K2),Augment2(ghz_val2,K2),QB(la_val2,K2),Augment2(ghz_val3,K2),QB(la_val3,K2)
ghz1,g=test_bound(number3)
Ghz1=Augment2(ghz1,K)
bound=dict()
#%%

#np.save('./Data/3-qubit/label_data_'+repr(number1)+'.npy', label_data)#增强后的数据
#np.save('./Data/3-qubit/label_'+repr(number1)+'.npy', label)#增强后的标签

#np.save('./Data/3-qubit/la_'+'K'+repr(K)+'_'+repr(number1)+'.npy', la_data)#增强后的数据
''' 
label_data=np.load('./Data/3-qubit/label_data_'+repr(number1)+'.npy')
label=np.load('./Data/3-qubit/label_'+repr(number1)+'.npy')
la_data=la_data=Augment(label_data, K)
'''
unlabel_data=Un_State2(number2,label_data,label)
un=Augment2(unlabel_data,K)
#np.save('./Data/3-qubit/un_'+'K'+repr(K)+'_'+repr(number1)+'_'+repr(number2)+'.npy', un)#增强后的无标签数据

#un=np.load('./Data/3-qubit/un_'+'K'+repr(3)+'_'+repr(number1)+'_'+repr(number2)+'.npy')
'''
Ghz1,Ghz2,Ghz3,la1,la2,la3=test_data(number3) 
Ghz1=Augment2(Ghz1, K2)
Ghz2=Augment2(Ghz2, K2)
Ghz3=Augment2(Ghz3, K2)
la1=QB(la1,K2)
la2=QB(la2,K2)
la3=QB(la3,K2)
np.save('./Data/test_data/a'+repr(t)+'_Ghz1_'+'K2_'+repr(number3)+'.npy', Ghz1)
np.save('./Data/test_data/a'+repr(t)+'_Ghz2_'+'K2_'+repr(number3)+'.npy', Ghz2)
np.save('./Data/test_data/a'+repr(t)+'_Ghz3_'+'K2_'+repr(number3)+'.npy', Ghz3)
np.save('./Data/test_data/a'+repr(t)+'_la1_'+'K2_'+repr(number3)+'.npy', la1)
np.save('./Data/test_data/a'+repr(t)+'_la2_'+'K2_'+repr(number3)+'.npy', la2)
np.save('./Data/test_data/a'+repr(t)+'_la3_'+'K2_'+repr(number3)+'.npy', la3)
'''

epoch=100#监督
Epoch=150#半监督
threshould=0.99
rratio=[]
iteration=30
start=-1
stop=1
x=np.arange(start,stop,abs(start-stop)/iteration)
y=Gauss_data(x)
select_number=[]
bestepoch=[]
bound0,bound1,bound2,bound3=0.15,0.25,0.4,0.6
bound=pd.DataFrame(np.zeros((iteration+1,4)),columns=["bound0","bound1","bound2","bound3"])
val_acc=pd.DataFrame(np.zeros((iteration+1,4)),columns=["val_3s","val_2s","val_e","val_acc"])

pmodel= tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=[8,8]),  
    tf.keras.layers.Dense(512, activation='relu' ),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(256, activation='relu' ),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(128, activation='relu' ),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(16, activation='relu' ),
    #tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(3, activation='softmax' )
    ])
pmodel.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
pmodel.fit(la_data,lA, epochs=epoch ,validation_split=0,shuffle=True ,batch_size = 2000)
Up=pmodel.predict(un,batch_size = 2000)
bas = [Up[k:k + K] for k in range(0, len(Up),K)]
ub_average=1/(K)*(np.sum(bas, axis=1))
ub=np.argmax(ub_average,axis=1)
#ub_average=Sharpen(ub_average,T=0.5)
index=np.argwhere(ub_average>threshould)[:,0]
print('选择样本数：',(K)*len(index))
select_number.append((K)*len(index))
index=Index(index,K)
ub=to_categorical(ub, num_classes=3)
ub=QB(ub,K)
ub=ub[index]
   
bb=(pmodel.predict(Ghz1))
bbas = [bb[k:k + K] for k in range(0, len(bb),K)]
bb_average=1/(K)*(np.sum(bbas, axis=1))
bbb=np.argmax(bb_average,axis=1)
bound.iloc[0,0]=g[np.argwhere(bbb==1)[0]]
bound.iloc[0,1]=g[np.argwhere(bbb==0)[-1]]
bound.iloc[0,2]=g[np.argwhere(bbb==2)[0]]
bound.iloc[0,3]=g[np.argwhere(bbb==1)[-1]]

acc1=pmodel.evaluate(ghz_val1,la_val1)[1]
acc2=pmodel.evaluate(ghz_val2,la_val2)[1]
acc3=pmodel.evaluate(ghz_val3,la_val3)[1]
val_acc.iloc[0,0]=acc1
val_acc.iloc[0,1]=acc2
val_acc.iloc[0,2]=acc3
val_acc.iloc[0,3]=(acc1+acc2+acc3)/3
#pmodel.save('./model_3qubit/a'+repr(t)+'_pmodel_'+'K'+repr(K)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5')


for jj in range(iteration): 
    alpha=y[jj]          #(1/((np.pi*0.4)**(1/2)))*(math.exp( (-x[jj]**2)/0.4 ))
    print('iteration=',jj)
    X=np.concatenate((la_data,un[index]), axis=0)
    Y=np.concatenate((lA,ub), axis=0)
    label_size=len(la_data)
    mmodel= tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(input_shape=[1,8,8]),  
        tf.keras.layers.Dense(512, activation='relu' ),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(256, activation='relu' ),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(128, activation='relu' ),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(16, activation='relu' ),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(3, activation='softmax' )
        ])
    def mycrossentropy(y_true, y_pred,L_S=label_size,lamda=alpha):#batch_size标签数据
           cc =tf.keras.losses.CategoricalCrossentropy()
           loss1 =cc(y_true[0:L_S],y_pred[0:L_S])
           loss2 =cc(y_true[L_S:],y_pred[L_S:] )
           loss=loss1+lamda*loss2
           return loss
    mmodel.compile(loss=mycrossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
    mmodel.fit(X,Y,epochs=Epoch ,validation_split=0,batch_size = len(Y))    
    
    Up=mmodel.predict(un,batch_size = len(un))
    bas = [Up[k:k + K] for k in range(0, len(Up),K)]
    ub_average=1/(K)*(np.sum(bas, axis=1))
    ub=np.argmax(ub_average,axis=1)
    #ub_average=Sharpen(ub_average,T=0.5)
    index=np.argwhere(ub_average>threshould)[:,0]
    index=Index(index,K)
    ub=to_categorical(ub, num_classes=3)
    ub=QB(ub,K)
    ub=ub[index]
    print('选择样本数：',len(index))
    select_number.append(len(index))
    
    bb=mmodel.predict(Ghz1)
    bbas = [bb[k:k + K] for k in range(0, len(bb),K)]
    bb_average=1/(K)*(np.sum(bbas, axis=1))
    bbb=np.argmax(bb_average,axis=1)
    bound.iloc[jj+1,0]=g[np.argwhere(bbb==1)[0]]
    bound.iloc[jj+1,1]=g[np.argwhere(bbb==0)[-1]]
    bound.iloc[jj+1,2]=g[np.argwhere(bbb==2)[0]]
    bound.iloc[jj+1,3]=g[np.argwhere(bbb==1)[-1]]
    acc1=mmodel.evaluate(ghz_val1,la_val1)[1]
    acc2=mmodel.evaluate(ghz_val2,la_val2)[1]
    acc3=mmodel.evaluate(ghz_val3,la_val3)[1]
    val_acc.iloc[jj+1,0]=acc1
    val_acc.iloc[jj+1,1]=acc2
    val_acc.iloc[jj+1,2]=acc3
    val_acc.iloc[jj+1,3]=(acc1+acc2+acc3)/3
    
    #mmodel.save('./model_3qubit/a'+repr(t)+'_mmodel_'+'KK'+repr(K)+'_'+repr(jj)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5')
    #val_acc.to_csv('./Result/3-qubit/a'+repr(t)+'_valacc_kK'+repr(K)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.csv')
#%%
'''
from tensorflow import keras
alpha=y[2]
label_size=len(la_data)
def mycrossentropy(y_true, y_pred,L_S=label_size,lamda=alpha):#batch_size标签数据
           cc =tf.keras.losses.CategoricalCrossentropy()
           loss1 =cc(y_true[0:L_S],y_pred[0:L_S])
           loss2 =cc(y_true[L_S:],y_pred[L_S:] )
           loss=loss1+lamda*loss2
           return loss
best=keras.models.load_model('./max_model3/5_mmodel_KK1_2_1000_5000_0.95.h5',custom_objects={'mycrossentropy': mycrossentropy})
ghz1,ghz2,ghz3,a1,a2,a3=test_data(number3) 
b1=pmodel.evaluate(ghz1,a1)[1]
b2=pmodel.evaluate(ghz2,a2)[1]
b3=pmodel.evaluate(ghz3,a3)[1]

'''
