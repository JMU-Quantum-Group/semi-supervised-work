# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 16:41:17 2022

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
data=pd.read_csv("C:/Users/WIN10/Desktop/plot_da3333.csv",error_bad_lines=False)
t=0
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
x_v1=np.concatenate((Ghz1,Ghz2,Ghz3), axis=0)
y_v1=np.concatenate((la1,la2,la3), axis=0)
#ghz1,ghz2,ghz3,a1,a2,a3=test_data(number3) 

#y=np.argmax(y_test,axis=1)

#su_model.evaluate(X_test,y_test,batch_size =600)[1]
#semi_model.evaluate(val_S,truth_S,batch_size =len(val_S))[1]
#%%绘制子图
nb_classes=3
fig = plt.figure(figsize=(12,8))
ylims=[0.6,0.8,0.9,0.9]
xlims_right=[0.4,0.15,0.15,0.08]
lw =1.5
mv=[500,150,150,150]
ms=9
for (plt_index,K,jj,number1,number2) in zip(range(1,4,2),KK[0:2],JJ[0:2],L[0:2],U[0:2]):
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
    ax1= fig.add_subplot(2,2,plt_index)
    ax2= fig.add_subplot(2,2,plt_index+1)
    axins1 = ax1.inset_axes((0.5, 0.5, 0.5, 0.4))
    axins2= ax2.inset_axes((0.5, 0.5, 0.5, 0.4))
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
 
    ax1.plot(fpr1["micro"], tpr1["micro"],
    label='supervised-micro-average (area = {0:0.3f})'
    ''.format(roc_auc1["micro"]),
    color='lightpink',  linewidth=lw)
    ax2.plot(fpr2["micro"], tpr2["micro"],
    label='semi-micro-average (area = {0:0.3f})'
    ''.format(roc_auc2["micro"]),
    color='lightpink', linestyle='-.',linewidth=lw)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    axins1.plot(fpr1["micro"], tpr1["micro"],
    color='lightpink',  linewidth=lw)
   
    axins2.plot(fpr2["micro"], tpr2["micro"],
    color='lightpink', linestyle='-.',marker='*', markevery=mv[0],markersize= ms,linewidth=lw)
    for i, color in zip(range(nb_classes), colors):
        ax1.plot(fpr1[i], tpr1[i], color=color, lw=lw,
        label='supervised-class {0} (area = {1:0.3f})'
        ''.format(i, roc_auc1[i]))  
        ax2.plot(fpr2[i], tpr2[i], color=color,linestyle=':', lw=lw,
        label='semi-class {0} (area = {1:0.3f})'
        ''.format(i, roc_auc2[i]))
        
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('l='+repr(number1)+', '+'u='+repr(number2))
        ax1.legend(loc="lower left")
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('l='+(number1)+', '+'u='+repr(number2))
        ax2.legend(loc="lower left")
        
        #plt.savefig("../images/ROC/ROC_3分类.png")
        axins1.plot(fpr1[i], tpr1[i], color=color, lw=lw)
        axins2.plot(fpr2[i], tpr2[i], color=color,linestyle=':', lw=lw,marker='o',markersize=ms-2, markerfacecolor='none',markevery=mv[i+1])
        # 设置放大区间
        zone_left = 0
        zone_right = xlims_right[plt_index-1]
        ylim0=ylims[plt_index-1]
        ylim1=1
        # 调整子坐标系的显示范围
        axins1.set_xlim(zone_left, zone_right)
        axins1.set_ylim(ylim0, ylim1)
        axins2.set_xlim(zone_left, zone_right)
        axins2.set_ylim(ylim0, ylim1)

plt.tight_layout(pad=1,h_pad=3.0,w_pad=3.0)  
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))


















