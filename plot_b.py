# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 21:43:10 2022

@author: zlf
"""
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("C:/Users/WIN10/Desktop/第二个工作/plot_da_b.csv",error_bad_lines=False)
a=[1/8,1/4,1/2]
fig= plt.figure(figsize=(12,12))
x = [0,1,2,3,4,5,6,7]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
N=[4,5,6,7]
y=[0,1,2,3,4,5,6,7]
ax={}
for i in range(4):
    da=data[i*8:(i+1)*8]
    ax["{0}".format(i+1)] = fig.add_subplot(2,2,i+1)
    ax["{0}".format(i+1)].set_ylim([0.17, 0.45])
    ax["{0}".format(i+1)].set_xticks(range(len(x)), x)
    y=da['b']
    y1=da['a5']
    y2=da['a6']
    ax["{0}".format(i+1)].plot(x,y, '-r', label='b='+repr(y.iloc[0]))
    ax["{0}".format(i+1)].plot(x,y2, color=colors[1],linestyle='-.', marker='s', markersize= 5,label='semi & mean='+repr(y2.describe().round(5)["mean"]))
    ax["{0}".format(i+1)].plot(x,y1, color=colors[0],linestyle=':',marker='*', markersize= 5,label='supervised & mean='+repr(y1.describe().round(5)["mean"]))
    ax["{0}".format(i+1)].legend()
    ax["{0}".format(i+1)].set_xlabel('Different training sets for n='+repr(N[i])+' and a=1/2')
    ax["{0}".format(i+1)].set_title('Learning bounds')
    ax["{0}".format(i+1)].set_ylabel('b')
    plt.show()






