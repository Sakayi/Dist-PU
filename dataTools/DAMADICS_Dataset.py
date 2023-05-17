# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:58:13 2023

@author: Song Keyu
"""


import numpy as np
import pandas as pd

# RawDataPath = r"D:\data\DAMADICS\Lublin_all_data"
import settings
RawDataPath = settings.DAMADICS_DATA_DIR


Vars = ["-",
    "P51_05",
    "P51_06",
    "T51_01",
    "F51_01", 
    "LC51_03CV" ,
    "LC51_03X" ,
    "LC51_03PV" ,
    "TC51_05" ,
    "T51_08" ,
    "D51_01" ,
    "D51_02" ,
    "F51_02" ,
    "PC51_01" ,
    "T51_06" ,
    "P51_03" ,
    "T51_07" ,
    "P57_03" ,
    "P57_04" ,
    "T57_03" ,
    "FC57_03PV" ,
    "FC57_03CV" ,
    "FC57_03X" ,
    "P74_00" ,
    "P74_01" ,
    "T74_00" ,
    "F74_00" ,
    "LC74_20CV" ,
    "LC74_20X" ,
    "LC74_20PV" ,
    "F74_30" ,
    "P74_30" ,
    "T74_30" ,]
RelatedVarId = {
    "Actuator1":[i for i in range(1,17)],
    "Actuator2":[i for i in range(17,23)],
    "Actuator3":[i for i in range(23,33)],
    "All":[i for i in range(1,33)],
    }


def PackRawData():
    '''
    Load txt files from RawDataPath.
    Save a concatenated whole Dataframe to DAMADICS.pkl in RawDataPath.
    '''
    from os.path import join
    from os import listdir
    import pickle
    
    files = listdir(RawDataPath)
    files = files[-3:]+files[:-3]
    DF_list = []
    for f in files:
        DF_list.append(pd.read_csv(join(RawDataPath,f),sep="\t"))
        DF_list[-1].columns=Vars
    raw_data = pd.concat(DF_list,ignore_index=True)
    with open(join(RawDataPath,"DAMADICS.pkl"),"wb") as output:
        pickle.dump(raw_data, output)


def LoadFullData():
    from os.path import join
    import pickle
    with open(join(RawDataPath,"DAMADICS.pkl"),"rb") as input:
        raw_data = pickle.load(input)
        return raw_data
    return None
    
def CreateTrainingSet(length = 12,step = 6):
    raw_data = LoadFullData()
    dr = raw_data[::30]
    sample_num = int( 14*(24*60*2) / step)
    X = np.empty(shape=(sample_num,length,32))
    for i in range(sample_num):
        X[i,:] = dr.iloc[i*step:i*step+length,1:]
    Y = np.zeros(shape=(sample_num,))
    Y[int( 3*(24*60*2) / step):int( 11*(24*60*2) / step)] = True
    return X,Y

def CreateTestSet(length = 12,step = 6):
    raw_data = LoadFullData()
    dr = raw_data[::30]
    sample_start = int((14)*(24*60*2))
    sample_test = int( ((25-14)*(24*60*2) -length)/ step)
    X = np.empty(shape=(sample_test,length,32))
    for i in range(sample_test):
        X[i,:] = dr.iloc[sample_start+i*step:sample_start+i*step+length,1:]
    Y = np.zeros(shape=(sample_test,))
    Y[int( 3*(24*60*2) / step):int( 11*(24*60*2) / step)] = True
    return X,Y


def CalcIndex(Day=0,Sec=50000,length=12,step=2):
    return int ((((2+Day)*24*60*60+Sec)/30-length)/step)


def HalfHalfTest():
    X,Y = CreateTrainingSet(length = 12,step = 2)
    Xt,Yt = CreateTestSet(length = 12,step = 2)
    from SelfAT_comparative import SelfAT
    from LSTM_regression_0322 import LSTM_regression
    model = SelfAT(
        LSTM_regression(18,12,32),
        momentum=0.8)
    model.fit(X,np.array([Y,1-Y]).T,True,Xt)
    y_pred=model.predict(Xt)
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Qt5Agg')
    plt.plot(y_pred[:,0])


def TrainingSet1109(U_ratio=0.6,length=12,step=1,actuator="Actuator3"):
    from os.path import join
    DF0 = pd.read_csv(join(RawDataPath,"09112001.txt"),sep="\t")
    DF0.columns=Vars
    Y0 = np.ones(shape=(DF0.shape[0],))
    Y0[57275:57550] = False
    Y0[58830:58930] = False
    Y0[58520:58625] = False
    Y0[60650:60700] = False  
    Y0[60870:60960] = False
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(DF0)
    dr = scaler.transform((DF0[::30])[1980:])
    y0 = (Y0[::30])[1980:]
    sample_num = int((dr.shape[0]-length)/step)
    X = np.empty(shape=(sample_num,length,len(RelatedVarId[actuator])))
    Y = np.ones(shape=(sample_num,))
    for i in range(sample_num):
        X[i,:,:] = dr[i*step:i*step+length,RelatedVarId[actuator]]
        Y[i] = not False in y0[i*step:i*step+length]
    Y[:int(len(Y)*U_ratio)] = False
    return X,Y,scaler

def TestSet1117(scaler,length=12,step=1,actuator="Actuator3"):
    
    from os.path import join
    DF1 = pd.read_csv(join(RawDataPath,"17112001.txt"),sep="\t")
    DF1.columns=Vars
    Y1 = np.ones(shape=(DF1.shape[0],))
    if actuator == "Actuator1" or actuator == "All":
        Y1[54600:54700] = False
        Y1[56670:56770] = False
    if actuator == "Actuator2" or actuator == "All":
        Y1[53780:53794] = False
        Y1[54193:54215] = False
        Y1[55482:55517] = False
        Y1[55977:56015] = False
        Y1[57030:57072] = False
    if actuator == "Actuator3" or actuator == "All":
        Y1[57475:57530] = False
        Y1[57675:57800] = False
        Y1[58150:58325] = False
     
    dr1 = scaler.transform(DF1[::30])
    y1 = Y1[::30]
    sample_num = int((dr1.shape[0]-length)/step)
    Xt = np.empty(shape=(sample_num,length,len(RelatedVarId[actuator])))
    Yt = np.zeros(shape=(sample_num,))
    for i in range(sample_num):
        Xt[i,:,:] = dr1[i*step:i*step+length,RelatedVarId[actuator]]
        Yt[i] = not False in y1[i*step:i*step+length]
    
    return Xt,Yt

if __name__ == "__main__":
    length = 10
    step = 1
    actuator = "Actuator3" ## Change this
    # actuator = "All"
    
    from os.path import join
    DF0 = pd.read_csv(join(RawDataPath,"09112001.txt"),sep="\t")
    DF0.columns=Vars
    Y0 = np.ones(shape=(DF0.shape[0],))
    Y0[57275:57550] = False
    Y0[58830:58930] = False
    Y0[58520:58625] = False
    Y0[60650:60700] = False  
    Y0[60870:60960] = False
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(DF0)
    dr = scaler.transform((DF0[::30])[1980:])
    y0 = (Y0[::30])[1980:]
    sample_num = int((dr.shape[0]-length)/step)
    X = np.empty(shape=(sample_num,length,len(RelatedVarId[actuator])))
    Y = np.ones(shape=(sample_num,))
    for i in range(sample_num):
        X[i,:,:] = dr[i*step:i*step+length,RelatedVarId[actuator]]
        Y[i] = not False in y0[i*step:i*step+length]
    Y[:int(len(Y)*0.60)] = False
    # Y = np.zeros(shape=(sample_num,))
    # Y[int((2100-1940)/step):] = True
    # import matplotlib.pyplot as plt
    # plt.plot(Y)
    # plt.plot(X[:,0,-5])
    
    DF1 = pd.read_csv(join(RawDataPath,"17112001.txt"),sep="\t")
    DF1.columns=Vars
    Y1 = np.ones(shape=(DF1.shape[0],))
    Y1[54600:54700] = False
    Y1[56670:56770] = False
    Y1[53780:53794] = False
    Y1[54193:54215] = False
    Y1[55482:55517] = False
    Y1[55977:56015] = False
    Y1[57030:57072] = False
    Y1[57475:57530] = False
    Y1[57675:57800] = False
    Y1[58150:58325] = False
     
    dr1 = scaler.transform(DF1[::30])
    y1 = Y1[::30]
    sample_num = int((dr1.shape[0]-length)/step)
    Xt = np.empty(shape=(sample_num,length,len(RelatedVarId[actuator])))
    Yt = np.zeros(shape=(sample_num,))
    for i in range(sample_num):
        Xt[i,:,:] = dr1[i*step:i*step+length,RelatedVarId[actuator]]
        Yt[i] = not False in y1[i*step:i*step+length]
    Yt[0:int(50000/30/step)] = True
    
    
    
    
    from LSTM_regression_0322 import LSTM_regression,LSTM_classifier
    from MLP_regression import MLP_regression,MLP_classifier
    
    MODEL_TYPE = "SatPU" ## Change this
    if MODEL_TYPE == "SatPU":
        from SatPU import SatPU
        # model = SatPU(
        #     MLP_regression(100,length,len(RelatedVarId[actuator])),
        #     momentum=0.8)
        model = SatPU(
            LSTM_regression(8,length,len(RelatedVarId[actuator])),
            momentum=0.9)
        model.fit(X,np.array([Y,1-Y]).T,True,Xt)
    elif MODEL_TYPE == "Baseline":
        from BaseLine import Baseline
        model = Baseline(
            MLP_regression(12,length,len(RelatedVarId[actuator])))
        model.fit(X,np.array([Y,1-Y]).T,True,Xt)
    elif MODEL_TYPE == "DeepDCR":
        from DeepDCR import DeepDCR
        model = DeepDCR(
            MLP_classifier(12,length,len(RelatedVarId[actuator])))
        model.fit(X,Y,Xt)
    y_pred=model.predict(Xt)
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Qt5Agg')
    plt.plot(y_pred[:,0])
    # plt.plot(Xt[:,0,-5])
    plt.plot(Yt)
    Accuracy  = np.sum((y_pred[:,1]>0.5) ^ (Yt==1)) / len(Yt)
    Precision = np.sum((y_pred[:,1]>0.5) & (Yt==0)) / np.sum(y_pred[:,1]>0.5)
    Recall    = np.sum((y_pred[:,1]>0.5) & (Yt==0)) / np.sum(Yt==0)
    F1        = 2.0* Precision * Recall / (Precision + Recall)
    print("%.2f"%(Accuracy*100))
    print("%.2f"%(Precision*100))
    print("%.2f"%(Recall*100))
    print("%.2f"%(F1*100))
    '''(L8N60)(P60%)
    97.56
    96.30
    27.37
    42.62
    '''
    '''(L8N80)(P60%)
    97.25
    78.57
    23.16
    35.77
    '''
    '''(L18N90)(P10%)
    95.81
    95.00
    24.36
    38.78
    '''
    
    
