import torch
from torchvision import transforms

from .PUdataset import PUdataset
from .cifar10 import get_cifar10, binarize_cifar_class
from .fmnist import get_mnist_fashion, binarize_mnist_fashion_class
from .alzheimer import get_ad, ad_transform
from .TEP_Dataset import LoadDataSet,ModifyTrainingData
from .DAMADICS_Dataset import TestSet1117,TrainingSet1109
import numpy as np

def create_dataset(dataset, datapath, args=None):
    if dataset == 'cifar-10':
        (X_train, Y_train), (X_test, Y_test) = get_cifar10(datapath)
        Y_train, Y_test = binarize_cifar_class(Y_train, Y_test)
        transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    elif dataset == 'fmnist':
        (X_train, Y_train), (X_test, Y_test) = get_mnist_fashion(datapath)
        Y_train, Y_test = binarize_mnist_fashion_class(Y_train, Y_test)
        transform = None
    elif dataset == 'alzheimer':
        (X_train, Y_train), (X_test, Y_test) = get_ad(datapath)
        transform = ad_transform
    elif dataset == "TEP":
        from os.path import join
        from settings import TEP_DATA_DIR
        F_ratio=args.TEP_Fault_Ratio
        P_ratio=args.TEP_Positive_Ratio
        x_train, y_train, x_test, y_test = LoadDataSet(join(TEP_DATA_DIR,"IDV(%d).pkl"%(1)))
        X_train,Y_train = ModifyTrainingData(x_train,y_train,4000,F_ratio,P_ratio)
        
        t_norm = 10000
        X_test = np.concatenate([x_test[:t_norm,:,:],x_test[-int(t_norm*F_ratio):,:,:]])
        Y_test = np.concatenate([y_test[:t_norm,:],y_test[-int(t_norm*F_ratio):,:]])
        Y_train = Y_train[:,0].T
        Y_test = Y_test[:,0].T
        print ("[TEP Dataset]")
        print ((X_train.shape,Y_train.shape,X_test.shape,Y_test.shape))
        transform = None
    elif dataset == "DAMADICS":
        X_train,Y_train,scaler = TrainingSet1109(1-args.TEP_Positive_Ratio,12,1,"Actuator3")
        X_test,Y_test = TestSet1117(scaler,12,1,"Actuator3")
        print ("[DAMADICS Dataset]")
        transform = None
        
    else:
        raise NotImplementedError("The dataset: {} is not defined!".format(dataset))

    return BCDataset(X_train, Y_train, transform), BCDataset(X_test, Y_test, transform)

def create_pu_dataset(dataset_train, num_labeled):
    return PUdataset(dataset_train.X, dataset_train.Y, num_labeled, dataset_train.transform)

class BCDataset(torch.utils.data.Dataset):
    """
    BCDataset - Supervised Binary Classification dataset

    members:
        X - features
        Y - labels
    """

    def __init__(self, X, Y, transform=None):
        super().__init__()
        self.X = X
        self.Y = Y
        self.transform = transform
    
    def __len__(self):
        return(len(self.Y))

    def __getitem__(self, index):
        img = self.X[index]
        if self.transform is not None:
            img = self.transform(img)
        return index, img, self.Y[index]