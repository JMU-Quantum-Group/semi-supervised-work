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
def Get_BiSep(number): #二可分
    BiS=np.zeros((8,8)).reshape(1,8,8)
    PHI_BC=Get_SpeEnt(number)
    for phi_BC in PHI_BC:
        a1=np.random.normal(0, 1)+np.random.normal(0, 1)*1j
        a2=np.random.normal(0, 1)+np.random.normal(0, 1)*1j
        phi_A=a1*e_0+a2*e_1
        Rho_A_BC=Q.tensor(Qobj(np.dot(phi_A,(phi_A.trans().conj()))),Qobj(phi_BC))
        Rho_A_BC=(1/(Rho_A_BC.tr()))*Rho_A_BC
        Rho_A_BC=np.array(Rho_A_BC)
        BiS=np.concatenate((BiS,Rho_A_BC.reshape(1,8,8)), axis=0)
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
def Get_SpeEnt(number):#2-qubit,1/3
    SpeE=np.zeros((4,4)).reshape(1,4,4)
    I2=np.eye(4)
    for i in range(int(number/2)):
        ghz_2=((1/np.sqrt(2))*(I2[0,:]+I2[3,:])).reshape(4,1)
        w_2=((1/np.sqrt(2))*(I2[1,:]+I2[2,:])).reshape(4,1)
        g=random.uniform(1/3,1)
        w=random.uniform(1/3,1)
        Rho_ghz=(1-g)*I2/4+g*(np.dot(ghz_2,(ghz_2.conj().T)))
        Rho_w=(1-w)*I2/4+w*(np.dot(w_2,(w_2.conj().T)))     
        SpeE=np.concatenate((SpeE,Rho_ghz.reshape(1,4,4),Rho_w.reshape(1,4,4)), axis=0)
    SpeE=SpeE[1:,:,:]
    return SpeE
    
    
def Get_GenEnt(number): #真正纠缠,最终得到6*number
    GenE=np.zeros((8,8)).reshape(1,8,8)
    for i in range(int(number/6)):
        ghz_3=((1/np.sqrt(2))*(I[0,:]+I[7,:])).reshape(8,1)
        w_3=((1/np.sqrt(3))*(I[1,:]+I[2,:]+I[3,:])).reshape(8,1)
        g=random.uniform(0.429,1)
        w=random.uniform(0.5,1)
        Rho_ghz=(1-g)*I/8+g*(np.dot(ghz_3,(ghz_3.conj().T)))
        Rho_w=(1-w)*I/8+w*(np.dot(w_3,(w_3.conj().T)))     
        GenE=np.concatenate((GenE,Rho_ghz.reshape(1,8,8),Rho_w.reshape(1,8,8)), axis=0)
        P=[1/2,2/5,1/5,1/10]
        Q=[0.556,0.44,0.31,0.27]
        for (p,q) in zip(P,Q):
            gw=random.uniform(q,1)
            Rho_ghz_w=(1-gw)*I/8+gw*(p*(np.dot(ghz_3,(ghz_3.conj().T)))+\
                                     (1-p)*(np.dot(w_3,(w_3.conj().T))))      
            GenE=np.concatenate((GenE,Rho_ghz_w.reshape(1,8,8)), axis=0)
    GenE=GenE[1:,:,:]
    label=np.zeros((len(GenE),3))
    label[:,2]=1
    return GenE,label
def Test_State(number):
    PS,PSL=Get_PerSep(number)
    BS,BSL=Get_BiSep(number)
    GE,GEL=Get_GenEnt(number)
    return PS,PSL,BS,BSL,GE,GEL
def GhzW_State():
    GhzW_05=np.zeros((8,8)).reshape(1,8,8)
    GhzW_04=np.zeros((8,8)).reshape(1,8,8)
    GhzW_02=np.zeros((8,8)).reshape(1,8,8)
    GhzW_01=np.zeros((8,8)).reshape(1,8,8)
    P=[1/2,2/5,1/5,1/10]
    ghz_3=((1/np.sqrt(2))*(I[0,:]+I[7,:])).reshape(8,1)
    w_3=((1/np.sqrt(3))*(I[1,:]+I[2,:]+I[3,:])).reshape(8,1)
    GW= np.arange(0,1,0.001)
    for gw in GW:
            gw=random.uniform(0,1)
            Rho_ghz_w0=(1-gw)*I/8+gw*(P[0]*(np.dot(ghz_3,(ghz_3.conj().T)))+\
                                    (1-P[0])*(np.dot(w_3,(w_3.conj().T))))  
            Rho_ghz_w1=(1-gw)*I/8+gw*(P[1]*(np.dot(ghz_3,(ghz_3.conj().T)))+\
                                    (1-P[1])*(np.dot(w_3,(w_3.conj().T))))  
            Rho_ghz_w2=(1-gw)*I/8+gw*(P[2]*(np.dot(ghz_3,(ghz_3.conj().T)))+\
                                    (1-P[2])*(np.dot(w_3,(w_3.conj().T))))  
            Rho_ghz_w3=(1-gw)*I/8+gw*(P[3]*(np.dot(ghz_3,(ghz_3.conj().T)))+\
                                    (1-P[3])*(np.dot(w_3,(w_3.conj().T))))  
            GhzW_05=np.concatenate((GhzW_05,Rho_ghz_w0.reshape(1,8,8)), axis=0)
            GhzW_04=np.concatenate((GhzW_04,Rho_ghz_w1.reshape(1,8,8)), axis=0)
            GhzW_02=np.concatenate((GhzW_02,Rho_ghz_w2.reshape(1,8,8)), axis=0)
            GhzW_01=np.concatenate((GhzW_01,Rho_ghz_w3.reshape(1,8,8)), axis=0)
    GhzW_05=GhzW_05[1:,:,:]
    GhzW_04=GhzW_04[1:,:,:]
    GhzW_02=GhzW_02[1:,:,:]
    GhzW_01=GhzW_01[1:,:,:]
    return GhzW_05,GhzW_04,GhzW_02,GhzW_01

def Un_State(number):
    UnS=np.zeros((8,8)).reshape(1,8,8)
    BS,BSL= Get_BiSep(int(number/4))
    PS,PSL=Get_PerSep(int(number/6))
    for i in range(int((number-(number/4+number/6))/6)):
        ghz_3=((1/np.sqrt(2))*(I[0,:]+I[7,:])).reshape(8,1)
        w_3=((1/np.sqrt(3))*(I[1,:]+I[2,:]+I[3,:])).reshape(8,1)
        g=random.uniform(0,1)
        w=random.uniform(0,1)
        Rho_ghz=(1-g)*I/8+g*(np.dot(ghz_3,(ghz_3.conj().T)))
        Rho_w=(1-w)*I/8+w*(np.dot(w_3,(w_3.conj().T)))     
        UnS=np.concatenate((UnS,Rho_ghz.reshape(1,8,8),Rho_w.reshape(1,8,8)), axis=0)
        P=[1/2,2/5,1/5,1/10]
        for p in P:
            gw=random.uniform(0,1)
            Rho_ghz_w=(1-gw)*I/8+gw*(p*(np.dot(ghz_3,(ghz_3.conj().T)))+\
                                    (1-p)*(np.dot(w_3,(w_3.conj().T))))      
            UnS=np.concatenate((UnS,Rho_ghz_w.reshape(1,8,8)), axis=0)
    UnS=UnS[1:,:,:]
    UnS=np.concatenate((UnS,BS,PS), axis=0)
    permutation=list(np.random.permutation(len(UnS)))
    UnS=UnS[permutation]
    return UnS
def Augment(data,M):#对无标签数据进行3*K次增强,truth,one-hot编码
    Ub_hut=np.zeros((8,8)).reshape(1,8,8)
    for ub in data:
            ub_hut= Unitary_trans(ub,M)
            Ub_hut=np.concatenate((Ub_hut,ub_hut), axis=0)
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
    PS,PSL=Get_PerSep(int((1/2)*number))
    BS,BSL=Get_BiSep(number)
    GE,GEL=Get_GenEnt(int(2*number))
    label_data=np.concatenate((PS,BS,GE), axis=0)
    label=np.concatenate((PSL,BSL,GEL), axis=0)
    permutation=list(np.random.permutation(len(label_data)))
    label_data=label_data[permutation]
    label=label[permutation]
    return label_data,label
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
    
#%%
Get_state3(50000)
#%%
bound_3=np.load('./pptmixer/bound_state3.npy')
number1=100
number2=20000
number3=2000
K=2     
label_data,label=get_labeldata(number1)
label_dataAug=Augment(label_data, K)
lA=QB(label,K)
unlabel_data=Un_State(number2)
unlabel_dataAug=Augment(unlabel_data,K)
#%%
Tps,Tpsl,Tbs,Tbsl,Tge,Tgel=Test_State(number3)
Tps=Augment(Tps, K)
Tbs=Augment(Tbs, K)
Tge=Augment(Tge, K)
Tpsl=QB(Tpsl,K)
Tbsl=QB(Tbsl,K)
Tgel=QB(Tgel,K)
#%%
epoch=50#监督
Epoch=100#半监督
rratio=[]
iteration=20
start=0.1
stop=1
x=np.arange(start,stop,abs(start-stop)/iteration)
select_number=[]
bestepoch=[]
LOSS_supervised=[]
acc=np.zeros((iteration+1,4))
pmodelloss=[]
#%%监督
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
for i in range(epoch):
    print('epoch=',i)
    h=pmodel.fit(label_dataAug,lA, epochs=1 ,validation_split=0,shuffle=True ,batch_size = 60)
    Up=pmodel.predict_classes(unlabel_dataAug,batch_size = 600)
    hloss=h.history['loss'][0]
    pmodelloss.append(hloss) 
pmodelbestepoch=np.argmin(pmodelloss)
bestmodel= tf.keras.models.Sequential(
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
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(3, activation='softmax' )
    ])
bestmodel.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
bestmodel.fit(label_dataAug,lA, epochs=pmodelbestepoch ,validation_split=0,shuffle=True ,batch_size = 50)
Up=bestmodel.predict(unlabel_dataAug,batch_size = 600)
bas = [Up[k:k + K] for k in range(0, len(Up),K)]
ub_average=1/(K)*(np.sum(bas, axis=1))
ub=np.argmax(ub_average,axis=1)
#ub_average=Sharpen(ub_average,T=0.5)
index=np.argwhere(ub_average>0.98)[:,0]
print('选择样本数：',(K)*len(index))
select_number.append((K)*len(index))
ratio=sum(ub[index])/len(ub[index])
rratio.append(ratio)
index=Index(index,K)
ub=to_categorical(ub, num_classes=3)
ub=QB(ub,K)
ub=ub[index]
bestepoch.append(pmodelbestepoch)
acc0=bestmodel.evaluate(Tps,Tpsl,batch_size =int(number3/2))[1]
acc1=bestmodel.evaluate(Tbs,Tbsl,batch_size =int(number3/2))[1]
acc2=bestmodel.evaluate(Tge,Tgel,batch_size =int(number3/2))[1]
acc[0,0]=acc0
acc[0,1]=acc1
acc[0,2]=acc2
acc[0,3]=(acc0+acc1+acc2)/3
#%%迭代更新伪标签
for jj in range(iteration): 
    iterationloss=[]
    alpha=x[jj]          #(1/((np.pi*0.4)**(1/2)))*(math.exp( (-x[jj]**2)/0.4 ))
    print('iteration=',jj)
    X=np.concatenate((label_dataAug,unlabel_dataAug[index]), axis=0)
    Y=np.concatenate((lA,ub), axis=0)
    label_size=len(label_dataAug)
    LOSS=[]
    mmodel= tf.keras.models.Sequential(
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
        m=mmodel.fit(X,Y, epochs=1 ,validation_split=0 ,batch_size = len(Y)) 
        mloss=m.history['loss'][0]
        iterationloss.append(mloss)     
    bepoch=np.argmin(iterationloss)
    bestepoch.append(bepoch)
    bbestmodel= tf.keras.models.Sequential(
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
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(3, activation='softmax' )
        ])
    bbestmodel.compile(loss=crossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
    bbestmodel.fit(X,Y, epochs=bepoch ,validation_split=0 ,batch_size = len(Y))
    acc0=bbestmodel.evaluate(Tps,Tpsl,batch_size =int(number3/2))[1]
    acc1=bbestmodel.evaluate(Tbs,Tbsl,batch_size =int(number3/2))[1]
    acc2=bbestmodel.evaluate(Tge,Tgel,batch_size =int(number3/2))[1]
    acc[jj+1,0]=acc0
    acc[jj+1,1]=acc1
    acc[jj+1,2]=acc2
    acc[jj+1,3]=(acc0+acc1+acc2)/3
    Up=bbestmodel.predict(unlabel_dataAug,batch_size = len(unlabel_dataAug))
    bas = [Up[k:k + K] for k in range(0, len(Up),K)]
    ub_average=1/(K)*(np.sum(bas, axis=1))
    ub=np.argmax(ub_average,axis=1)
    #ub_average=Sharpen(ub_average,T=0.5)
    index=np.argwhere(ub_average>0.98)[:,0]
    ratio=sum(ub[index])/len(ub[index])
    rratio.append(ratio)
    index=Index(index,K)
    ub=to_categorical(ub, num_classes=3)
    ub=QB(ub,K)
    ub=ub[index]
    print('选择样本数：',len(index))
    select_number.append(len(index))
bestmodel.save('./MY_MODEL/bestmodel_K3_2000.h5')
np.save('./Result/3_qubit/bestepoch_K'+repr(K)+'_'+repr(number1)+'.npy', bestepoch)#选择的优良epoch
np.save('./Result/3_qubit/testacc_K'+repr(K)+'_'+repr(number1)+'.npy', acc)#伪标签精度
np.save('./Result/3_qubit/select_number_K'+repr(K)+'_'+repr(number1)+'.npy', select_number)#选择加入损失的样本数
np.save('./Result/3_qubit/Ratio_K'+repr(K)+'_'+repr(number1)+'.npy', rratio)
#%%
GhzW_05,GhzW_04,GhzW_02,GhzW_01=GhzW_State()
cc=bestmodel.predict_classes(GhzW_005)

GhzW_005=Augment(GhzW_05, 1)
#%%



