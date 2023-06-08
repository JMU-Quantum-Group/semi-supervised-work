# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 21:44:58 2022

@author: zlf
"""

#使用numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("C:/Users/WIN10/Desktop/plot_da22.csv",error_bad_lines=False)
semi_acc=data['semi_acc']
suk_acc=data['suk_acc']
su_acc=data['su_acc']
l=data['l']
u=data['u']
T= dict()
i=0
K1=2
K2=4
#%% width 设置条形的宽度
barWidth=0.2
r1 = np.arange(len(semi_acc[0:4]))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
fig=plt.figure(figsize=(12,4.5))
ax1 = fig.add_subplot(1,2,1)
ax1.bar(r3,semi_acc[0:4],width=barWidth,label='semi & '+'K='+repr(K1))
ax1.bar(r2,suk_acc[0:4],width=barWidth,label='supervised & '+'K='+repr(K1))
ax1.bar(r1,su_acc[0:4],width=barWidth,label='supervised')
# 设置数据标签
ax1.set_ylabel('Accuracy')
ax1.set_xticks([r + barWidth for r in range(len(semi_acc[0:4]))],['l=500','l=1000','l=2000','l=4000'])
ax2 = fig.add_subplot(1,2,2)
ax2.bar(r3,semi_acc[5:9],width=barWidth,label='semi & '+'K='+repr(K2))
ax2.bar(r2,suk_acc[5:9],width=barWidth,label='supervised & '+'K='+repr(K2))
ax2.bar(r1,su_acc[0:4],width=barWidth,label='supervised')
# 设置数据标签

ax2.set_ylabel('Accuracy')
ax2.set_xticks([r + barWidth for r in range(len(semi_acc[5:9]))],['l=500','l=1000','l=2000','l=4000'])
ax1.legend()
ax2.legend()
plt.show()
#%%

width = 0.16  # 柱状图的宽度，可以根据自己的需求和审美来改
x = np.arange(len(semi_acc[0:4]))  # 标签位置
r1 = x - width*2
r2 =x - width+0.01
r3 = x + 0.02
r4=x + width+ 0.03
r5=x + width*2 + 0.04
fig=plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(1,2,1)
ax1.bar(r1,su_acc[0:4],width=width,label='supervised')
ax1.bar(r2,suk_acc[0:4],width=width,label='supervised & '+'K='+repr(K1))
ax1.bar(r3,semi_acc[0:4],width=width,label='semi & '+'K='+repr(K1))
ax1.bar(r4,suk_acc[5:9],width=width,label='supervised & '+'K='+repr(K2))
ax1.bar(r5,semi_acc[5:9],width=width,label='semi & '+'K='+repr(K2))
# 设置数据标签
ax1.set_ylabel('Accuracy')
ax1.set_ylim([0.0, 1.0])
ax1.set_xticks([r  for r in range(len(semi_acc[0:4]))],['l=500','l=1000','l=2000','l=4000'])
ax1.set_title('Classification Accuracy')
ax1.legend(bbox_to_anchor=(1.02, 0.4), loc=3, borderaxespad=0)
plt.show()
#%%
width = 0.15
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
# 柱状图的宽度，可以根据自己的需求和审美来改

semi_eacc=data['semi_eacc']
semi_sacc=data['semi_sacc']
suk_eacc=data['suk_eacc']
suk_sacc=data['suk_sacc']
su_eacc=data['su_eacc']
su_sacc=data['su_sacc']
x = np.arange(len(semi_acc[0:4]))  # 标签位置
r1 = x - width*4
r2 =x - 3*width
r3 = x - 2*width
r4=x -width
r5=x 
r6=x + 1*width 
r7=x + 2*width 
r8=x + 3*width 
r9=x + 4*width 

fig3=plt.figure(figsize=(18,4))
ax3 = fig3.add_subplot(1,3,1)
ax3.bar(r1,su_sacc[0:4],width=width,color=colors[1],edgecolor="black",label='supervised-class 0')
ax3.bar(r2,suk_sacc[0:4],width=width,color=colors[1],hatch='.',edgecolor="black",label='supervised & K-class 0')
ax3.bar(r3,semi_sacc[0:4],width=width,color=colors[1],hatch='/',edgecolor="black",label='semi & K-class 0')

ax3.bar(r4,su_eacc[0:4],width=width,color=colors[0],edgecolor="black",label='supervised-class 1')
ax3.bar(r5,suk_eacc[0:4],width=width,color=colors[0],hatch='.',edgecolor="black",label='supervised & K=-class 1')
ax3.bar(r6,semi_eacc[0:4],width=width,color=colors[0],hatch='/',edgecolor="black",label='semi & K-class 1')

ax4 = fig3.add_subplot(1,3,2)
ax4.bar(r1,su_sacc[0:4],width=width,color=colors[1],edgecolor="black",label='supervised-class 0')
ax4.bar(r2,suk_sacc[5:9],width=width,color=colors[1],hatch='.',edgecolor="black",label='supervised & K-class 0')
ax4.bar(r3,semi_sacc[5:9],width=width,color=colors[1],hatch='/',edgecolor="black",label='semi & K-class 0')

ax4.bar(r4,su_eacc[0:4],width=width,color=colors[0],edgecolor="black",label='supervised-class 1')
ax4.bar(r5,suk_eacc[5:9],width=width,color=colors[0],hatch='.',edgecolor="black",label='supervised & K-class 1')
ax4.bar(r6,semi_eacc[5:9],width=width,color=colors[0],hatch='/',edgecolor="black",label='semi & K-class 1')

# 设置数据标签
ax3.set_ylim([0.0, 1.0])
ax3.set_ylabel('Accuracy')
ax3.set_xticks([r  for r in range(len(semi_acc[0:4]))],['l=500','l=1000','l=2000','l=4000'])
ax3.set_title('Classification Accuracy on Test Dataset'+'(K='+repr(K1)+')')
ax4.set_ylim([0.0, 1.0])
ax4.set_ylabel('Accuracy')
ax4.set_xticks([r  for r in range(len(semi_acc[0:4]))],['l=500','l=1000','l=2000','l=4000'])
ax4.set_title('Classification Accuracy on Test Dataset'+'(K='+repr(K2)+')')
ax4.legend(bbox_to_anchor=(1.02, 0.4), loc=3, borderaxespad=0)
plt.show()
























