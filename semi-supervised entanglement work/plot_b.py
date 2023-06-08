# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 21:43:10 2022

@author: zlf
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("C:/Users/好梦难追/Desktop/semi-supervised entanglement work/plotb.csv",error_bad_lines=False,header=None)
a=[1/8,1/4,1/2]
fig= plt.figure(figsize=(12,12))
x = [0,1,2,3,4,5,6,7]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
N=[4,5,6,7]
y=[0,1,2,3,4,5,6,7]
ax={}
for i in range(4):
    da=data[i*9:(i+1)*9]
    ax["{0}".format(i+1)] = fig.add_subplot(2,2,i+1)
    ax["{0}".format(i+1)].set_ylim([0.1, 0.5])
    ax["{0}".format(i+1)].set_xticks(range(len(x)), x)
    y=da[0][1:9]
    y1=da[1][1:9]
    y2=da[2][1:9]
    ax["{0}".format(i+1)].plot(x,y, '-r',marker='d', markersize= 5, label='b')
    ax["{0}".format(i+1)].plot(x,y1, color=colors[0],linestyle=':',marker='*', markersize= 5,label='semi & '+'a='+repr(1/8))
    ax["{0}".format(i+1)].plot(x,y2, color=colors[1],linestyle='-.', marker='s', markersize= 5,label='supervised & '+'a='+repr(1/8))
    ax["{0}".format(i+1)].legend(loc="lower right")
    ax["{0}".format(i+1)].set_xlabel('Different training sets for n='+repr(N[i]))
    ax["{0}".format(i+1)].set_title('Learning bounds by SSL and SL')
    ax["{0}".format(i+1)].set_ylabel('b')
    plt.show()






