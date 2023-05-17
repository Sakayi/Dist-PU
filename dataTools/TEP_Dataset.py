# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:07:54 2022

@author: Song Keyu
"""
import numpy as np
import pandas as pd
import pickle
from os.path import join
from settings import TEP_DATA_DIR

### Save Load两个方法直接读取准备好的pkl，不需要从原始数据开始准备
def SaveDataSet(x_train,y_train,x_test,y_test,file="dataset.pkl"):
    import pickle
    with open(file,"wb") as output:
        pickle.dump(x_train, output)
        pickle.dump(y_train, output)
        pickle.dump(x_test, output)
        pickle.dump(y_test, output)


def LoadDataSet(file="dataset.pkl"):
    import pickle
    with open(file,"rb") as input:
        x_train = pickle.load(input)
        y_train = pickle.load(input)
        x_test = pickle.load(input)
        y_test = pickle.load(input)
        return x_train,y_train,x_test,y_test
    return None,None,None,None

def DataFrameFromPKL(file_name:str):
    with open(join(TEP_DATA_DIR,file_name),"rb") as input:
        return pickle.load(input)
    
def DataFrameToPKL(DF:pd.DataFrame,file_name:str):
    with open(join(TEP_DATA_DIR,file_name),"wb") as output:
        pickle.dump(DF, output)

### TEP数据集 生成序列识别问题实验数据
### subset 取值0~1 用于分割出小部分数据
def TEP_Windowed_DataSet(fault_number = 1,
                         input_length=12,
                         step=12,
                         subset = 1.0,
                         ambigous = 0.0):
    '''
    
    Parameters
    ----------
    fault_number : int (0~20), optional
        TEP fault case ID. Default = 1.
    input_length : int (1~500), optional
        输入长度.  Default = 12.
    step : int (1~500), optional
        切片距离.  Default = 12.
    subset : float (0.0~1.0), optional
        只利用部分数据时保留的比例.  Default = 1.0.
    ambigous : float (0.0~1.0), optional
        模糊一定比例的正常训练数据.  Default = 0.0.
        
    Returns
    -------
    x_train : numpy.ndarray
    y_train : numpy.ndarray
    x_test : numpy.ndarray
    y_test : numpy.ndarray
    
    Example
    -------
    x_train,y_train,x_test,y_test = TEP_Windowed_DataSet(1,12,12,0.1)
    
    x_train,y_train,x_test,y_test=TEP_Windowed_DataSet(
        fault_number=1,
        input_length=12,
        step=12,
        subset=0.1,
        ambigous=0.8)
    '''
    from math import floor
    
    ### 原始数据
    DF0 = DataFrameFromPKL("TEP_FaultFree_Training.pkl")
    DF0 = DF0.iloc[:floor(DF0.shape[0]*subset),:]
    DF1 = DataFrameFromPKL("TEP_Faulty_Training.pkl")
    DF1 = DF1[DF1["faultNumber"]==fault_number]
    DF1 = DF1.iloc[:floor(DF1.shape[0]*subset),:]
    DF_Test = DataFrameFromPKL("TEP_Faulty_Testing.pkl")
    DF_Test = DF_Test[DF_Test["faultNumber"]==fault_number]
    DF_Test = DF_Test.iloc[:floor(DF_Test.shape[0]*subset),:]
    DF_Test_0 = DataFrameFromPKL("TEP_FaultFree_Testing.pkl")
    DF_Test_0 = DF_Test_0.iloc[:floor(DF0.shape[0]*subset),:]
    
    ### 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(pd.concat((DF0.iloc[:,3:],DF1.iloc[:,3:])))
    
    ### 正常状态训练数据
    Train_Run_Length = 500 # TEP数据集训练样本长度
    Fault_Inducted_Time = 20 # TEP数据集训练样本故障引入时间
    normal_sample_count = floor((DF0.shape[0]-input_length)/step)
    train_runs = floor(DF1["simulationRun"].max())
    abnorml_samples_each = floor((Train_Run_Length-input_length)/step)
    x_train = np.empty(shape=(
        normal_sample_count+train_runs*abnorml_samples_each,
        input_length,
        DF0.shape[1]-3))
    y_train = np.empty(shape=(
        normal_sample_count+train_runs*abnorml_samples_each,
        2))
    for i in range(normal_sample_count):
        x_train[i,:,:] = scaler.transform(
            DF0.iloc[i*step:i*step+input_length,
                    3:])
        if i < ambigous * normal_sample_count:
            y_train[i,:]   = [0.0,1.0]
        else: 
            y_train[i,:]   = [1.0,0.0]
        
    ### 异常状态训练数据
    for run in range(train_runs):
        Sample = DF1[DF1["simulationRun"]==run+1]
        for i in range(abnorml_samples_each):
            train_id = normal_sample_count + run*abnorml_samples_each + i
            # print (i)
            # print (Sample.iloc[i*step:i*step+input_length,3:].shape)
            x_train[train_id,:,:] = scaler.transform(
                Sample.iloc[i*step:i*step+input_length,
                            3:])
            ### 用窗口最后一点的状态标识该切片是否异常
            if i*step+input_length < Fault_Inducted_Time and ambigous == 0.0:
                y_train[train_id,:] = [1.0,0.0]
            else :
                y_train[train_id,:] = [0.0,1.0]
    
    
    ### 测试集数据
    Test_Run_Length = 960 # TEP数据集测试样本长度
    Fault_Inducted_Time_T = 100 # TEP数据集测试样本故障引入时间
    test_runs = floor(DF_Test["simulationRun"].max())
    normal_sample_count_t = floor((DF_Test_0.shape[0]-input_length)/step)
    abnorml_samples_each_t = floor((Test_Run_Length-input_length)/step)
    x_test = np.empty(shape=(
        normal_sample_count_t + test_runs * abnorml_samples_each_t,
        input_length,
        DF_Test.shape[1]-3))
    y_test = np.empty(shape=(
        normal_sample_count_t + test_runs*abnorml_samples_each_t,
        2))
    
    ### 正常状态测试集数据
    for i in range(normal_sample_count_t):
        x_test[i,:,:] = scaler.transform(
            DF_Test_0.iloc[i*step:i*step+input_length,
                    3:])
        if i < ambigous * normal_sample_count_t:
            y_test[i,:]   = [0.0,1.0]
        else: 
            y_test[i,:]   = [1.0,0.0]
    
    ### 异常状态测试集数据
    for run in range(test_runs):
        Sample = DF_Test[DF_Test["simulationRun"]==run+1]
        for i in range(abnorml_samples_each_t):
            test_id = normal_sample_count_t + run*abnorml_samples_each_t + i
            x_test[test_id,:,:] = scaler.transform(
                Sample.iloc[i*step:i*step+input_length,
                            3:])
            ### 用窗口最后一点的状态标识该切片是否异常
            if i*step+input_length < Fault_Inducted_Time_T:
                y_test[test_id,:] = [1.0,0.0]
            else :
                y_test[test_id,:] = [0.0,1.0]
    
    return x_train,y_train,x_test,y_test

def SaveClearData():
    import settings
    from os.path import join
    for f in range(1,21):
        x_train,y_train,x_test,y_test=TEP_Windowed_DataSet(
            fault_number=f,
            input_length=12,
            step=12,
            subset=1.0,
            ambigous=0.0)
        SaveDataSet(x_train, y_train, x_test, y_test,file = join(settings.TEP_DATA_DIR,"IDV(%d).pkl"%(f)))

def GetSavedTrainingData(data_folder,fault_number):
    from os.path import exists,join
    file_path = join(data_folder,"IDV(%d).pkl"%(fault_number))
    if not exists(file_path):
        x_train,y_train,x_test,y_test=TEP_Windowed_DataSet(
            fault_number=fault_number,
            input_length=12,
            step=12,
            subset=1.0,
            ambigous=0.0)
        SaveDataSet(x_train, y_train, x_test, y_test,file = file_path)
        return x_train,y_train,x_test,y_test
    else:
        return LoadDataSet(file_path)

def ModifyTrainingData(x_train,y_train,Total=4000,F_ratio=0.4,P_ratio=0.1):
    '''
    Parameters
    ----------
    x_train : numpy.ndarray
        shape(40832, 12, 52).
    y_train : numpy.ndarray
        shape(40832, 2).
    Total : TYPE, optional
        total samples. The default is 4000.
    F_ratio : TYPE, optional
        Fault ratio. The default is 0.4.
    P_ratio : TYPE, optional
        Correctly labeled Normal Samples(ratio). The default is 0.1.

    Returns
    -------
    X, Y.

    '''
    if not(F_ratio + P_ratio <= 1.0 and F_ratio>=0.0 and P_ratio >= 0.0):
        return None,None
    F_samples = int(F_ratio * Total)
    N_samples = int(P_ratio  * Total)
    X = np.concatenate([x_train[:Total-F_samples,:,:],x_train[-F_samples:,:,:]],axis=0)
    Y = np.concatenate([np.ones((N_samples,1)),np.zeros((Total-N_samples,1))])
    Y = np.concatenate([Y,1-Y],axis=1)
    return X,Y
    

### TEP数据集 生成单点识别问题实验数据 经过标准化
### subset 取值0~1 用于分割出小部分数据
def TEP_DataSet(fault_number = 1,subset = 1.0):
    DF0 = DataFrameFromPKL("TEP_FaultFree_Training.pkl")
    DF0 = DF0.iloc[:int(DF0.shape[0]*subset),3:]
    DF1 = DataFrameFromPKL("TEP_Faulty_Training.pkl")
    DF1 = DF1[DF1["faultNumber"]==fault_number]
    DF1 = DF1.iloc[:int(DF1.shape[0]*subset),3:]
    DF_Test = DataFrameFromPKL("TEP_Faulty_Testing.pkl")
    DF_Test = DF_Test[DF_Test["faultNumber"]==fault_number]
    DF_Test = DF_Test.iloc[:int(DF_Test.shape[0]*subset),3:]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    x_train=scaler.fit_transform(pd.concat((DF0,DF1)))
    y_train=np.concatenate((
        np.repeat([[1.0,0.0]],DF0.shape[0],axis=0),
        np.repeat([[0.5,0.5]],DF1.shape[0],axis=0)),
        axis=0)
    
    x_test=scaler.transform(DF_Test)
    y_test=np.tile(np.concatenate((
        np.repeat([[1.0,0.0]],100,axis=0),
        np.repeat([[0.0,1.0]],960-100,axis=0)),
        axis=0),(int(x_test.shape[0]/960),1))
    return x_train,y_train,x_test,y_test

### TEP数据集 转存Matlab的.mat格式
def TEP_TO_MAT(fault_number = 1,save_folder = r"D:\data\标签分布\TEP"):
    x_train,y_train,x_test,y_test = TEP_DataSet(fault_number)
    
    from scipy.io import savemat
    from os.path import join
    savemat(join(save_folder,"IDV(%d).mat"%(fault_number)),
            {'trainFeature':x_train,
             'trainDistribution':y_train,
             'testFeature':x_test,
             'testDistribution':y_test})

### TEP数据集 生成单点识别问题实验数据 经过标准化
def TEP_DataSet_(fault_number = 1):
    DF0 = DataFrameFromPKL("TEP_FaultFree_Training.pkl")
    DF1 = DataFrameFromPKL("TEP_Faulty_Training.pkl")
    DF_Test = DataFrameFromPKL("TEP_Faulty_Testing.pkl")
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train=scaler.fit_transform(pd.concat([DF0,DF1[DF1["faultNumber"]==fault_number]])[DF0.columns[3:]])
    y_train=np.concatenate((
        np.repeat([[1.0,0.0]],DF0.shape[0],axis=0),
        np.repeat([[0.5,0.5]],DF1[DF1["faultNumber"]==fault_number].shape[0],axis=0)),
        axis=0)
    x_test=scaler.transform(DF_Test[DF_Test["faultNumber"]==fault_number][DF_Test.columns[3:]])
    y_test=np.tile(np.concatenate((
        np.repeat([[1.0,0.0]],100,axis=0),
        np.repeat([[0.0,1.0]],960-100,axis=0)),
        axis=0),(int(x_test.shape[0]/960),1))
    return x_train,y_train,x_test,y_test

