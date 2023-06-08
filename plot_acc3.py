# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 21:43:10 2022

@author: zlf
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("C:/Users/WIN10/Desktop/第二个工作/max_result/3-qubit/plot_da_s.csv",error_bad_lines=False)

semi_e=data['semi_e']
semi_2s=data['semi_2s']
semi_3s=data['semi_3s']

su_e=data['su_e']
su_2s=data['su_2s']
su_3s=data['su_3s']



semig_e=data['semig_e']
semig_2s=data['semig_2s']
semig_3s=data['semig_3s']

sug_e=data['sug_e']
sug_2s=data['sug_2s']
sug_3s=data['sug_3s']


l=data['l']
u=data['u']
K1=1
width = 0.15
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
# 柱状图的宽度，可以根据自己的需求和审美来改
x = np.arange(len(semi_3s[0:4]))  # 标签位置
r1 = x - width*3
r2 =x - 2*width
r3 = x - 1*width
r4=x
r5=x + 1*width 
r6=x + 2*width 


fig=plt.figure(figsize=(12,4))
rect1 = [0.05, 0.10, 0.35, 0.8] # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
rect2 = [0.45, 0.10, 0.35, 0.80]
ax1 =  plt.axes(rect1)
ax1.bar(r1,su_3s[0:4],width=width,color=colors[0],edgecolor="black",label='supervised-class 0')
ax1.bar(r2,semi_3s[0:4],width=width,color=colors[0],hatch='/',edgecolor="black",label='semi-class 0')
ax1.bar(r3,su_2s[0:4],width=width,color=colors[1],edgecolor="black",label='supervised-class 1')
ax1.bar(r4,semi_2s[0:4],width=width,color=colors[1],hatch='/',edgecolor="black",label='semi-class 1')
ax1.bar(r5,su_e[0:4],width=width,color=colors[2],edgecolor="black",label='supervised-class 2')
ax1.bar(r6,semi_e[0:4],width=width,color=colors[2],hatch='/',edgecolor="black",label='semi-class 2' )
# 设置数据标签
ax1.set_ylim([0.0, 1.0])
ax1.set_ylabel('Accuracy')
ax1.set_xticks([r  for r in range(len(semi_3s[0:4]))],['l=20','l=50','l=100','l=200'])
ax1.set_title('Classification Accuracy on GHZ class states')


ax2 =  plt.axes(rect2)
ax2.bar(r1,sug_3s[0:4],width=width,color=colors[0],edgecolor="black",label='supervised-class 0')
ax2.bar(r2,semig_3s[0:4],width=width,color=colors[0],hatch='/',edgecolor="black",label='semi-class 0')
ax2.bar(r3,sug_2s[0:4],width=width,color=colors[1],edgecolor="black",label='supervised-class 1')
ax2.bar(r4,semig_2s[0:4],width=width,color=colors[1],hatch='/',edgecolor="black",label='semi-class 1')
ax2.bar(r5,sug_e[0:4],width=width,color=colors[2],edgecolor="black",label='supervised-class 2')
ax2.bar(r6,semig_e[0:4],width=width,color=colors[2],hatch='/',edgecolor="black",label='semi-class 2' )
# 设置数据标签
ax2.set_ylim([0.0, 1.0])
ax2.set_ylabel('Accuracy')
ax2.set_xticks([r  for r in range(len(semi_3s[0:4]))],['l=20','l=50','l=100','l=200'])
ax2.set_title('Classification Accuracy on GHZ states')
ax2.legend(bbox_to_anchor=(1.02, 0.4), loc=3, borderaxespad=0)
plt.show()
#%%

fig2= plt.figure(figsize=(12,5))
semi_acc=data['semi_acc']
su_acc=data['su_acc']


semig_acc=data['semig_acc']
sug_acc=data['sug_acc']
L=[20,50,100,200]
x = range(len(L))
K=8

ax2 = fig2.add_subplot(1,2,1)
ax2.plot(x,semi_acc[0:4], color=colors[0],linestyle=':',marker='*', markersize= 5,label='semi & '+'K='+repr(K))
ax2.plot(x,su_acc[0:4], color=colors[1],linestyle='-.', marker='s', markersize= 5,label='supervised & '+'K='+repr(K))

ax2.set_xticks(range(len(L)), L)
ax2.set_ylim([0.85, 1.00])
ax2.legend(loc="lower right")
ax2.set_xlabel('l')
ax2.set_title('Classification Accuracy on GHZ class states')
ax2.set_ylabel('Accaracy')


ax4 = fig2.add_subplot(1,2,2)
ax4.plot(x,semig_acc[0:4], color=colors[0],linestyle=':',marker='*', markersize= 5,label='semi & '+'K='+repr(K))
ax4.plot(x,sug_acc[0:4], color=colors[1],linestyle='-.', marker='s', markersize= 5,label='supervised & '+'K='+repr(K))
ax4.set_xticks(range(len(L)), L)
ax4.set_ylim([0.9, 1.00])
ax4.set_title('Classification Accuracy on GHZ states')
ax4.legend(loc="lower right")
ax4.set_xlabel('l')
ax4.set_ylabel('Accaracy')



