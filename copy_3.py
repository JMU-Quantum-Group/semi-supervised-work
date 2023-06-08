import numpy as np
import qutip as Q
from qutip import *
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
import pandas as pd
from keras.utils.np_utils import to_categorical
## 可分 纯态生成
e_0=Qobj(np.array([1, 0]))
e_1=Qobj(np.array([0, 1]))
I=np.eye(8)
def Unitary_trans(rho,K): #data,array,(,8,8),K次局部酉变换
    rho.reshape(8,8)
    UR=np.zeros((8,8)).reshape(1,8,8)
    ii=np.arange(0, 3)
    for k in range(K):
        u={}
        para=np.random.randint(-10**8, 10**8, size=(3, 3))
        for (gama,beta,delta,i) in zip(para[:,0],para[:,1],para[:,2],ii):
            u["u{0}".format(i)]=np.exp(1j*np.random.randint(-10**8, 10**8))\
                *np.dot(np.diag([np.exp(-1j*beta/2),np.exp(1j*beta/2)]),\
                        np.array([[np.cos(gama/2),-np.sin(gama/2)],[np.sin(gama/2),np.cos(gama/2)]]),\
                            np.diag([np.exp(-1j*delta/2),np.exp(1j*delta/2)]))
        Rho_augment=Q.tensor(Qobj(u['u0']),Qobj(u['u1']),Qobj(u['u2']))*rho*\
            ((Q.tensor(Qobj(u['u0']),Qobj(u['u1']),Qobj(u['u2']))).conj().trans()) 
        UR=np.concatenate((UR,Rho_augment.reshape(1,8,8)), axis=0)
    UR=UR[1:,:,:]
    return UR

#可分态凸组合
def Convex_Combination(data,B):#one-hot编码,b次凸组合
    #p_model=load_model('./MY_MODEL/Pmodel.h5')  
    X_hut=np.zeros((8,8)).reshape(1,8,8)
    permutation1 = list(np.random.permutation(len(data)))
    se1 = data[permutation1]
    se2=data
    for i in range(B):
        for (x1,x2) in zip(se1,se2):       
            bb=np.random.dirichlet(np.ones(2), size=1).reshape(2,1)
            x_hut=(bb[0]*x1+bb[1]*x2).reshape(1,8,8)
            X_hut=np.concatenate((X_hut,x_hut), axis=0) 
        se1=X_hut[1+i*len(se2):(i+1)*len(se2)+1,:,:]
        permutation2= list(np.random.permutation(len(se1)))
        se1=se1[permutation2]
    Sn_hut =X_hut[1:,:,:]
    return Sn_hut
def Get_PerSep(number): #全可分
    PerS=np.zeros((8,8)).reshape(1,8,8)
    for k in range(number):
        m=np.random.randint(2,15)
        P = np.random.dirichlet(np.ones(m))
        Rho=np.zeros((8,8))
        for p in P:
            a0=np.random.normal(0, 1,[2,2])
            a1=np.random.normal(0, 1,[2,2])
            b0=np.random.normal(0, 1,[2,2])
            b1=np.random.normal(0, 1,[2,2])
            c0=np.random.normal(0, 1,[2,2])
            c1=np.random.normal(0, 1,[2,2])
            rho_A=(np.dot(a0+1j*a1,((a0+1j*a1).T).conjugate()))
            rho_A=(1/rho_A.trace())*rho_A
            rho_B=(np.dot(b0+1j*b1,((b0+1j*b1).T).conjugate()))
            rho_B=(1/rho_B.trace())*rho_B
            rho_C=(np.dot(c0+1j*c1,((c0+1j*c1).T).conjugate()))
            rho_C=(1/rho_C.trace())*rho_C
            rho=p*(Q.tensor(Qobj(rho_A),Qobj(rho_B),Qobj(rho_C)))
            rho=np.array(rho)
            Rho=Rho+rho
        PerS=np.concatenate((PerS,Rho.reshape(1,8,8)), axis=0)
    PerS=PerS[1:,:,:]
    label=np.zeros((len(PerS),3))
    label[:,0]=1
    return PerS,label
def ACB_Kron(AC,B):
    R=np.zeros((8,8),dtype='complex_')
    R[0:4,0:4]=np.kron(B,AC[0:2,0:2])
    R[0:4,4:8]=np.kron(B,AC[0:2,2:4])
    R[4:8,0:4]=np.kron(B,AC[2:4,0:2])
    R[4:8,4:8]=np.kron(B,AC[2:4,2:4])
    return R
def Get_BiSep(number): #二可分
    BiS=np.zeros((8,8)).reshape(1,8,8)
    m=np.random.randint(2,15)
    PHI_BC=Get_Ent(int(3*m*number))
    j=0
    while j<3*m*number:
        G= np.random.dirichlet(np.ones(3))
        Rho_AB_C=np.zeros((8,8))
        Rho_AC_B=np.zeros((8,8))
        Rho_A_BC=np.zeros((8,8))
        P = np.random.dirichlet(np.ones(m))
        for p in P:
            M=np.random.normal(0, 1,[2,2])
            N=np.random.normal(0, 1,[2,2])
            H=np.dot(M+1j*N,((M+1j*N).T).conjugate()) #H      
            rho_1=(1/(np.trace(H)))*H #密度矩阵
            rho1=p*np.kron(rho_1,PHI_BC[j])
            rho1=np.array(rho1)
            Rho_AB_C=Rho_AB_C+rho1
            j+=1
        P = np.random.dirichlet(np.ones(m))
        for p in P:
            M=np.random.normal(0, 1,[2,2])
            N=np.random.normal(0, 1,[2,2])
            H=np.dot(M+1j*N,((M+1j*N).T).conjugate()) #H      
            rho_2=(1/(np.trace(H)))*H #密度矩阵
            rho2=p*ACB_Kron(PHI_BC[j],rho_2)
            rho2=np.array(rho2)
            Rho_AC_B=Rho_AC_B+rho2
            j+=1
        P = np.random.dirichlet(np.ones(m))
        for p in P:
            M=np.random.normal(0, 1,[2,2])
            N=np.random.normal(0, 1,[2,2])
            H=np.dot(M+1j*N,((M+1j*N).T).conjugate()) #H      
            rho_3=(1/(np.trace(H)))*H #密度矩阵
            rho3=p*np.kron(rho_3,PHI_BC[j])
            rho3=np.array(rho3)
            Rho_A_BC=Rho_A_BC+rho3
            j+=1
        Rho=  G[0]*Rho_AB_C+ G[1]*Rho_AC_B+G[2]*Rho_A_BC
        BiS=np.concatenate((BiS,Rho.reshape(1,8,8)), axis=0)
    BiS=BiS[1:,:,:]
    label=np.zeros((len(BiS),3))
    label[:,1]=1
    return BiS,label

def Get_Ent(number): #2-qubit 纠缠量子态
    L1=0
    R_entangle=np.zeros((4,4)).reshape(1,4,4)
    y=np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
    while L1<number:
       M=np.random.normal(0, 1,[4,4])
       N=np.random.normal(0, 1,[4,4])
       H=np.dot(M+1j*N,((M+1j*N).T).conjugate()) #H      
       Rho=(1/(np.trace(H)))*H #密度矩阵
       Alpha=np.dot(np.dot(np.dot(Rho,y),Rho.conjugate()),y)#
       eigenvalue,featurevector=np.linalg.eig(Alpha)
       eigenvalue.sort()
       C_Rho=(np.max([0,(eigenvalue[3])**(1/2)-\
                      (eigenvalue[2])**(1/2)-(eigenvalue[1])**(1/2)-(eigenvalue[0])**(1/2)]))
       Rho=Rho.reshape(1,4,4)
       if C_Rho>0.0000000000001:#判断量子矩阵可分纠缠
            R_entangle=np.concatenate((R_entangle,Rho), axis=0)
            L1+=1
    R_entangle=R_entangle[1:,:,:]
    return R_entangle

     
def Ghz(number): 
    Ghz1=np.zeros((8,8)).reshape(1,8,8)
    Ghz2=np.zeros((8,8)).reshape(1,8,8)
    Ghz3=np.zeros((8,8)).reshape(1,8,8)
    ghz_3=((1/np.sqrt(2))*(I[0,:]+I[7,:])).reshape(8,1)
    G=np.linspace(0,1,number)
    G1=G[np.where(G<0.2)]
    G2=G[np.where((G>=0.2) &(G<0.429))]
    G3=G[np.where(G>=0.429)]
    for (g1,g2,g3) in zip(G1,G2,G3):
        ghz1=(1-g1)*I/8+g1*(np.dot(ghz_3,(ghz_3.conj().T)))
        ghz2=(1-g2)*I/8+g2*(np.dot(ghz_3,(ghz_3.conj().T)))
        ghz3=(1-g3)*I/8+g3*(np.dot(ghz_3,(ghz_3.conj().T)))
        Ghz1=np.concatenate((Ghz1,ghz1.reshape(1,8,8)), axis=0)
        Ghz2=np.concatenate((Ghz2,ghz2.reshape(1,8,8)), axis=0)
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
def Un_State(number):
    UnS=np.zeros((8,8)).reshape(1,8,8)
    BS,BSL= Get_BiSep(int(number/7))
    PS,PSL=Get_PerSep(int(number/3))
    for i in range(int(number-(number/3+number/7))):
        M=np.random.normal(0, 1,[8,8])
        N=np.random.normal(0, 1,[8,8])
        H=np.dot(M+1j*N,((M+1j*N).T).conjugate()) #H      
        Rho=(1/(np.trace(H)))*H #密度矩阵
        UnS=np.concatenate((UnS,Rho.reshape(1,8,8)), axis=0)
    UnS=UnS[1:,:,:]
    UnS=np.concatenate((UnS,BS,PS), axis=0)
    permutation=list(np.random.permutation(len(UnS)))
    UnS=UnS[permutation]
    return UnS
def Augment(data,M):#对无标签数据进行3*K次增强,truth,one-hot编码
    Ub_hut=np.zeros((8,8)).reshape(1,8,8)
    for ub in data:
            ub_hut= Unitary_trans(ub,M-1)
            Ub_hut=np.concatenate((Ub_hut,ub.reshape(1,8,8),ub_hut), axis=0)
    Ub_hut=Ub_hut[1:,:,:]
    return Ub_hut
def Index(index,K):
    index1=np.zeros((1,))
    ii=np.ones((len(index1)))
    indexx1=(K)*index
    for k in range(K):
        indexk=indexx1+k*ii
        index1=np.concatenate((index1,indexk), axis=0)
    index1=np.sort(index1,axis=0)
    index1=index1.astype(np.int64)
    index1=index1[1:]
    return index1
def QB(qb,K):
    #qb = to_categorical(qb, num_classes=None)
    Yb_hut=np.zeros((K*len(qb),3)) 
    for k in range(K):
        Yb_hut[np.arange(k,len(Yb_hut),K),:]=qb
    return Yb_hut
def get_labeldata(number):
    PS,PSL=Get_PerSep(number)
    BS,BSL=Get_BiSep(number)
    bound,bla=get_bound()
    label_data=np.concatenate((PS,BS,bound[0:number]), axis=0)
    label=np.concatenate((PSL,BSL,bla[0:number]), axis=0)
    permutation=list(np.random.permutation(len(label_data)))
    label_data=label_data[permutation]
    label=label[permutation]
    return label_data,label
def test_data(number):
    PS,PSL=Get_PerSep(number)
    BS,BSL=Get_BiSep(number)
    bound,bla=get_bound()
    bound=bound[len(bound)-number:len(bound)]
    bla=bla[len(bound)-number:len(bound)]
    
    return PS,BS,bound,PSL,BSL,bla
def Get_state3(number): #
    R=np.zeros((8,8)).reshape(1,8,8)
    L1=0
    while L1<number:
        M=np.random.normal(0, 1,[8,8])
        N=np.random.normal(0, 1,[8,8])
        H=np.dot(M+1j*N,((M+1j*N).T).conjugate()) #H      
        Rho=(1/(np.trace(H)))*H #密度矩阵
        np.savetxt('./pptmixer/bound_state3/'+'state_real'+repr(L1+1)+'.txt',Rho.real, fmt="%.10f",delimiter=' ')
        np.savetxt('./pptmixer/bound_state3/'+'state_imag'+repr(L1+1)+'.txt',Rho.imag, fmt="%.10f",delimiter=' ')
        R=np.concatenate((R,Rho.reshape(1,8,8)), axis=0)
        L1+=1
    R=R[1:,:,:]
    np.save('./pptmixer/bound_state3.npy',R)
def get_bound():
    bound_3=np.load('./pptmixer/bound_state3.npy')
    la_b=pd.read_table("./pptmixer/Detect_bound.txt",sep=' ',header=None)
    la_b=la_b.loc[:,2]
    la_b=np.array(la_b)
    itermindex=np.argwhere(la_b==1)
    itermindex=itermindex.reshape(len(itermindex),)
    bound_3en=bound_3[itermindex]
    en=np.load('./pptmixer/3-qubit_ent.npy')
    bound_3en2=en[:,:,:,0]+1j*en[:,:,:,1]  
    bound=np.concatenate((bound_3en,bound_3en2[0:100]), axis=0)
    permutation=list(np.random.permutation(len(bound)))
    bound=bound[permutation]
    label=np.zeros((len(bound),3))
    label[:,2]=1
    return bound,label

    
        

#%%
N=[50,100,500]
M=[10000,30000,60000]
number3=500
K=3
number1=100
number2=10000
#for number1 in N:
 #   for number2 in M:
label_data,label=get_labeldata(number1)
label_dataAug=Augment(label_data, K)
lA=QB(label,K)
unlabel_data=Un_State(number2)
unlabel_dataAug=Augment(unlabel_data,K)
PS,BS,bound,PSL,BSL,bla =test_data(number3)
APS=Augment(PS,K)
ABS=Augment(BS,K)
Abound=Augment(bound,K)
APSL=QB(PSL,K)
ABSL=QB(BSL,K)
Abla=QB(bla,K)
#%%
epoch=50#监督
Epoch=110#半监督
rratio=[]
iteration=30
start=0.1
stop=1
x=np.arange(start,stop,abs(start-stop)/iteration)
select_number=[]
bestepoch=[]
acc=np.zeros((iteration+1,4))
#%%
pmodel= tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=[8,8]),  
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
    tf.keras.layers.Dense(3, activation='softmax' )
    ])
pmodel.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])

h=pmodel.fit(label_dataAug,lA, epochs=100 ,validation_split=0,shuffle=True ,batch_size = 60)
Up=pmodel.predict_classes(unlabel_dataAug,batch_size = 600) 
Up=pmodel.predict(unlabel_dataAug,batch_size = 600)
bas = [Up[k:k + K] for k in range(0, len(Up),K)]
ub_average=1/(K)*(np.sum(bas, axis=1))
ub=np.argmax(ub_average,axis=1)
#ub_average=Sharpen(ub_average,T=0.5)
index=np.argwhere(ub_average>0.98)[:,0]
print('选择样本数：',(K)*len(index))
select_number.append((K)*len(index))
index=Index(index,K)
ub=to_categorical(ub, num_classes=3)
ub=QB(ub,K)
ub=ub[index]
PS,BS,bound,PSL,BSL,bla
acc0=pmodel.evaluate(APS,APSL)[1]
acc1=pmodel.evaluate(ABS,ABSL)[1]
acc2=pmodel.evaluate(Abound,Abla)[1]
acc[0,0]=acc0
acc[0,1]=acc1
acc[0,2]=acc2
acc[0,3]=(acc0+acc1+acc2)/3
#%%

for jj in range(iteration): 
    iterationloss=[]
    alpha=x[jj]          #(1/((np.pi*0.4)**(1/2)))*(math.exp( (-x[jj]**2)/0.4 ))
    print('iteration=',jj)
    X=np.concatenate((label_dataAug,unlabel_dataAug[index]), axis=0)
    Y=np.concatenate((lA,ub), axis=0)
    label_size=len(label_dataAug)
    mmodel= tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(input_shape=[1,8,8]),  
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
        tf.keras.layers.Dense(3, activation='softmax' )
        ])
    def crossentropy(y_true, y_pred,L_S=label_size,lamda=alpha):#batch_size标签数据
           cc =tf.keras.losses.CategoricalCrossentropy()
           loss1 =cc(y_true[0:L_S],y_pred[0:L_S])
           loss2 =cc(y_true[L_S:],y_pred[L_S:] )
           loss=loss1+lamda*loss2
           return loss
    mmodel.compile(loss=crossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
    for ii in range(Epoch):
        print('ii=',ii)
        mmodel.fit(X,Y,epochs=1 ,validation_split=0,batch_size = len(Y))  
    acc0=mmodel.evaluate(APS,APSL)[1]
    acc1=mmodel.evaluate(ABS,ABSL)[1]
    acc2=mmodel.evaluate(Abound,Abla)[1]
    acc[jj+1,0]=acc0
    acc[jj+1,1]=acc1
    acc[jj+1,2]=acc2
    acc[jj+1,3]=(acc0+acc1+acc2)/3
    Up=mmodel.predict(unlabel_dataAug,batch_size = len(unlabel_dataAug))
    bas = [Up[k:k + K] for k in range(0, len(Up),K)]
    ub_average=1/(K)*(np.sum(bas, axis=1))
    ub=np.argmax(ub_average,axis=1)
    #ub_average=Sharpen(ub_average,T=0.5)
    index=np.argwhere(ub_average>0.98)[:,0]
    index=Index(index,K)
    ub=to_categorical(ub, num_classes=3)
    ub=QB(ub,K)
    ub=ub[index]
    print('选择样本数：',len(index))
    select_number.append(len(index))
mmodel.save('./MY_MODEL/model_'+'K',repr(K-1)+'_'+repr(number1)+repr(number2)+'.h5')
np.save('./Result/3_qubit/testacc_K'+repr(K-1)+'_'+repr(number1)+repr(number2)+'.npy', acc)#伪标签精度
np.save('./Result/3_qubit/select_number_K'+repr(K-1)+'_'+repr(number1)+repr(number2)+'.npy', select_number)#选择加入损失的样本数
#%%Ghz1,Ghz2,Ghz3,la1,la2,la3
Ghz1,Ghz2,Ghz3,la1,la2,la3,G=Ghz(2000)
ghz_acc1=  pmodel.evaluate(Ghz1,la1)[1]
ghz_acc2=  pmodel.evaluate(Ghz2,la2)[1]
ghz_acc3=  pmodel.evaluate(Ghz3,la3)[1]       
