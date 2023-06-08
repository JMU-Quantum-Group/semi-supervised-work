# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 17:26:05 2022

@author: zlf
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:04:36 2022

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
import fu_3_qubit
from fu_3_qubit import *
from sklearn.preprocessing import label_binarize
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from itertools import cycle
from scipy import interp
import time
start_time = time.time()
data=pd.read_csv("C:/Users/WIN10/Desktop/第二个工作/max_result/3-qubit/plot_da_s.csv",error_bad_lines=False)
t=4
threshould=0.99
KK=data['K']
L=data['l']
U=data['u']
JJ=data['iteration']
iteration=30
start=-1
stop=1
x=np.arange(start,stop,abs(start-stop)/iteration)
y=Gauss_data(x)
number3=6000

#%%
Ghz1=np.load('./Data/test_data/a5_Ghz1_'+'K2_'+repr(number3)+'.npy')
Ghz2=np.load('./Data/test_data/a5_Ghz2_'+'K2_'+repr(number3)+'.npy')
Ghz3=np.load('./Data/test_data/a5_Ghz3_'+'K2_'+repr(number3)+'.npy')
la1=np.load('./Data/test_data/a5_la1_'+'K2_'+repr(number3)+'.npy')
la2=np.load('./Data/test_data/a5_la2_'+'K2_'+repr(number3)+'.npy')
la3=np.load('./Data/test_data/a5_la3_'+'K2_'+repr(number3)+'.npy')
ghz1,ghz2,ghz3,a1,a2,a3=test_data(number3) 
x_v1=np.concatenate((Ghz1,Ghz2,Ghz3,ghz1,ghz2,ghz3), axis=0)
y_v1=np.concatenate((la1,la2,la3,a1,a2,a3), axis=0)
#ghz1,ghz2,ghz3,a1,a2,a3=test_data(number3) 

#y=np.argmax(y_test,axis=1)

#su_model.evaluate(X_test,y_test,batch_size =600)[1]
#semi_model.evaluate(val_S,truth_S,batch_size =len(val_S))[1]
#%%绘制子图
nb_classes=3
fig = plt.figure(figsize=(12,9))
ylims=[0.6,0.8,0.9,0.9]
xlims_right=[0.4,0.15,0.15,0.08]
lw =1.5
mv=[100,150,200]
ms=7
for (plt_index,K,jj,number1,number2) in zip(range(1,5),KK[0:4],JJ[0:4],L[0:4],U[0:4]):
    alpha=y[jj]
    label_size=(K)*number1
    def mycrossentropy(y_true, y_pred,L_S=label_size,lamda=alpha):#batch_size标签数据
        cc =tf.keras.losses.CategoricalCrossentropy()
        loss1 =cc(y_true[0:L_S],y_pred[0:L_S])
        loss2 =cc(y_true[L_S:],y_pred[L_S:] )
        loss=loss1+lamda*loss2
        return loss
    
    model_su = keras.models.load_model('./max_model3/a'+repr(t)+'_pmodel_K'+repr(K)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5')    
    model_semi=keras.models.load_model('./max_model3/a'+repr(t)+'_mmodel_'+'KK'+repr(K)+'_'+repr(jj)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5',custom_objects={'mycrossentropy': mycrossentropy})
    ax = fig.add_subplot(2,2,plt_index)
    axins = ax.inset_axes((0.5, 0.5, 0.5, 0.4))
    Y_pred1 = model_su.predict(x_v1)
    Y_pred2 = model_semi.predict(x_v1)
    y_v1 = [np.argmax(y) for y in y_v1]
    # Binarize the output
    y_v1 = label_binarize(y_v1, classes=[i for i in range(nb_classes)])

    fpr1 = dict()
    tpr1 = dict()
    roc_auc1 = dict()
    fpr2 = dict()
    tpr2 = dict()
    roc_auc2 = dict()
    thresholds_keras= dict()
    for i in range(nb_classes):
        fpr1[i], tpr1[i], thresholds_keras[i] = roc_curve(y_v1[:, i], Y_pred1[:, i])
        roc_auc1[i] = auc(fpr1[i], tpr1[i])
        # Compute micro-average ROC curve and ROC area
        fpr1["micro"], tpr1["micro"], _ = roc_curve(y_v1.ravel(), Y_pred1.ravel())
        roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])
       
        fpr2[i], tpr2[i], _ = roc_curve(y_v1[:, i], Y_pred2[:, i])
        roc_auc2[i] = auc(fpr2[i], tpr2[i])
        fpr2["micro"], tpr2["micro"], _ = roc_curve(y_v1.ravel(), Y_pred2.ravel())
        roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])
     
        # First aggregate all false positive rates
        # Then interpolate all ROC curves at this points
    colors = cycle(['#9467bd', '#ff7f0e', '#2ca02c'])
    linestyles=[':', '--','-.']
    markers=['*','s','d']
    for i, color in zip(range(nb_classes), colors):
        mv2=int(len(fpr1[i])/2)
        
        ax.plot(fpr2[i], tpr2[i], color=color,linestyle=linestyles[i],lw=lw,
        label='semi-class {0} (area = {1:0.3f})'
        ''.format(i, roc_auc2[i]))
        ax.plot(fpr1[i], tpr1[i], color=color, lw=lw,
        label='supervised-class {0} (area = {1:0.3f})'
        ''.format(i, roc_auc1[i]))  
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('l='+repr(number1)+', '+'u='+repr(number2))
        ax.legend(loc="lower left")
        #plt.savefig("../images/ROC/ROC_3分类.png")
        axins.plot(fpr1[i], tpr1[i], color=color,  lw=lw)
       # axins.plot(fpr2[i], tpr2[i], color=color,marker=markers[i], markevery=mv[i],markerfacecolor='none',markersize= 6, linestyle=linestyles[i], lw=lw)
        axins.plot(fpr2[i], tpr2[i], color=color, linestyle=linestyles[i], lw=lw)
        # 设置放大区间
        zone_left = 0
        zone_right = xlims_right[plt_index-1]
        ylim0=ylims[plt_index-1]
        ylim1=1
        # 调整子坐标系的显示范围
        axins.set_xlim(zone_left, zone_right)
        axins.set_ylim(ylim0, ylim1)

plt.tight_layout(pad=1,h_pad=3.0,w_pad=3.0)  
plt.show()
#%%
ylims2=[0.6,0.8,0.9,0.9]
xlims_right2=[0.4,0.15,0.15,0.15]
fig2 = plt.figure(figsize=(12,9))
for (plt_index,K,jj,number1,number2) in zip(range(1,5),KK[0:4],JJ[0:4],L[0:4],U[0:4]):
    alpha=y[jj]
    label_size=(K)*number1
    def mycrossentropy(y_true, y_pred,L_S=label_size,lamda=alpha):#batch_size标签数据
        cc =tf.keras.losses.CategoricalCrossentropy()
        loss1 =cc(y_true[0:L_S],y_pred[0:L_S])
        loss2 =cc(y_true[L_S:],y_pred[L_S:] )
        loss=loss1+lamda*loss2
        return loss
    
    model_su = keras.models.load_model('./max_model3/a'+repr(t)+'_pmodel_K'+repr(K)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5')    
    model_semi=keras.models.load_model('./max_model3/a'+repr(t)+'_mmodel_'+'KK'+repr(K)+'_'+repr(jj)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5',custom_objects={'mycrossentropy': mycrossentropy})
    ax = fig2.add_subplot(2,2,plt_index)
    axins = ax.inset_axes((0.5, 0.5, 0.5, 0.4))
    Y_pred1 = model_su.predict(x_v1)
    Y_pred2 = model_semi.predict(x_v1)
    y_v1 = [np.argmax(y) for y in y_v1]
    # Binarize the output
    y_v1 = label_binarize(y_v1, classes=[i for i in range(nb_classes)])

    fpr1 = dict()
    tpr1 = dict()
    roc_auc1 = dict()
    fpr2 = dict()
    tpr2 = dict()
    roc_auc2 = dict()
    thresholds_keras= dict()
    for i in range(nb_classes):
        fpr1[i], tpr1[i], thresholds_keras[i] = roc_curve(y_v1[:, i], Y_pred1[:, i])
        roc_auc1[i] = auc(fpr1[i], tpr1[i])
        # Compute micro-average ROC curve and ROC area
        fpr1["micro"], tpr1["micro"], _ = roc_curve(y_v1.ravel(), Y_pred1.ravel())
        roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])
       
        fpr2[i], tpr2[i], _ = roc_curve(y_v1[:, i], Y_pred2[:, i])
        roc_auc2[i] = auc(fpr2[i], tpr2[i])
        fpr2["micro"], tpr2["micro"], _ = roc_curve(y_v1.ravel(), Y_pred2.ravel())
        roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])
     
        # First aggregate all false positive rates
    all_fpr1 = np.unique(np.concatenate([fpr1[i] for i in range(nb_classes)]))
        # Then interpolate all ROC curves at this points
    mean_tpr1 = np.zeros_like(all_fpr1)

    all_fpr2 = np.unique(np.concatenate([fpr2[i] for i in range(nb_classes)]))
    mean_tpr2 = np.zeros_like(all_fpr2)
    for i in range(nb_classes):
        mean_tpr1 += np.interp(all_fpr1, fpr1[i], tpr1[i])
        # Finally average it and compute AUC
        mean_tpr1 /= nb_classes
        fpr1["macro"] = all_fpr1
        tpr1["macro"] = mean_tpr1
        roc_auc1["macro"] = auc(fpr1["macro"], tpr1["macro"])
        # Plot all ROC curves
        
        
        mean_tpr2 += np.interp(all_fpr2, fpr2[i], tpr2[i])
        # Finally average it and compute AUC
        mean_tpr2 /= nb_classes
        fpr2["macro"] = all_fpr2
        tpr2["macro"] = mean_tpr2
        roc_auc2["macro"] = auc(fpr2["macro"], tpr2["macro"])
    
    
   
    ax.plot(fpr2["micro"], tpr2["micro"],
    label='semi-micro-average (area = {0:0.3f})'
    ''.format(roc_auc2["micro"]),
    color='cyan', linestyle='--',linewidth=lw)
    
    ax.plot(fpr1["micro"], tpr1["micro"],
    label='supervised-micro-average (area = {0:0.3f})'
    ''.format(roc_auc1["micro"]),
    color='lightpink',  linewidth=lw)
    
    axins.plot(fpr1["micro"], tpr1["micro"],
    color='lightpink',  linewidth=lw)
    
    axins.plot(fpr2["micro"], tpr2["micro"],
    color='cyan', linestyle='--',linewidth=lw)  
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('l='+repr(number1)+', '+'u='+repr(number2))
    ax.legend(loc="lower left")
    #plt.savefig("../images/ROC/ROC_3分类.png")
    # 设置放大区间
    zone_left = 0
    zone_right = xlims_right2[plt_index-1]
    ylim0=ylims[plt_index-1]
    ylim1=1
    # 调整子坐标系的显示范围
    axins.set_xlim(zone_left, zone_right)
    axins.set_ylim(ylim0, ylim1)
print("--- %s seconds ---" % (time.time() - start_time))















