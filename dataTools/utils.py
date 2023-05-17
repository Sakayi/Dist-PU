# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:44:36 2023

@author: Song Keyu
"""
import numpy as np


def TrainingPlot(result_mat,y_true,plot=True):
    '''
    example
    ------
    A,P,R,F = TrainingPlot(model.result_valid,Yt[:1])
    '''
    Epochs      = result_mat.shape[0]
    Accuracy    = np.zeros((Epochs,))
    Precision   = np.zeros((Epochs,))
    Recall      = np.zeros((Epochs,))
    F1          = np.zeros((Epochs,))
    for i in range(Epochs):
        Accuracy[i]  = np.sum((result_mat[i,:]>0.5) ^ (y_true==0)) / len(y_true)
        Precision[i] = np.sum((result_mat[i,:]>0.5) & (y_true==1)) / np.sum(result_mat[i,:]>0.5)
        Recall[i]    = np.sum((result_mat[i,:]>0.5) & (y_true==1)) / np.sum(y_true==1)
        F1[i]        = 2.0* Precision[i] * Recall[i] / (Precision[i] + Recall[i])
    
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(Accuracy   ,label="Accuracy")
        plt.plot(Precision  ,label="Precision")
        plt.plot(Recall     ,label="Recall")
        plt.plot(F1         ,label="F1")
        plt.xlabel("epoch")
        plt.ylim(0,1)
        plt.legend()
        
    return Accuracy,Precision,Recall,F1

def TrainingPlot_TF(result_mat,y_true,plot=True):
    '''
    example
    ------
    TN,TP,FN,FP = TrainingPlot_TF(model.result_valid,Yt[:,1])
    '''
    Epochs      = result_mat.shape[0]
    TN = np.zeros((Epochs,))
    TP = np.zeros((Epochs,))
    FN = np.zeros((Epochs,))
    FP = np.zeros((Epochs,))
    for i in range(Epochs):
        TN[i]  = np.sum((result_mat[i,:]<0.5)  & (y_true==0))
        TP[i]  = np.sum((result_mat[i,:]>=0.5) & (y_true==1))
        FN[i]  = np.sum((result_mat[i,:]<0.5)  & (y_true==1))
        FP[i]  = np.sum((result_mat[i,:]>=0.5) & (y_true==0))
        
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(TN,label="TN")
        plt.plot(TP,label="TP")
        plt.plot(FN,label="FN")
        plt.plot(FP,label="FP")
        plt.xlabel("epoch")
        plt.ylim(0,1)
        plt.legend()
    return TN,TP,FN,FP


def GetTestY(fault_number=1,F_ratio=0.2,t_norm = 10000):
    from TEP_Dataset import LoadDataSet,ModifyTrainingData
    x_train,y_train,x_test,y_test = LoadDataSet("../../data/IDV(%d).pkl"%(fault_number))
    
    # Xt = np.concatenate([x_test[:t_norm,:,:],x_test[-int(t_norm*F_ratio):,:,:]])
    Yt = np.concatenate([y_test[:t_norm,:],y_test[-int(t_norm*F_ratio):,:]])
    
    return Yt

    
def ModelTitle(name = "SelfAT",F_ratio=0.2,P_ratio=0.4,fault_number=1):
    return "%s(%d)[F%d%%P%d%%]"%(name,fault_number,int(F_ratio*100),int(P_ratio*100))

def LoadResults(folder):
    import os
    import pickle
    with open(os.path.join(r"../../models/",folder,"validation.pkl"),"rb") as input:
        result = pickle.load(input)
        return result
    return None

'''
Example
-------
Result = LoadResults(ModelTitle("DeepDCR",0.2,0.75,1))
Yt = GetTestY(1,0.2)
A,P,R,F = TrainingPlot(Result,Yt)
print("%.2f"%(A[-1]*100))
print("%.2f"%(P[-1]*100))
print("%.2f"%(R[-1]*100))
print("%.2f"%(F[-1]*100))
'''