# ## import modules
import argparse
import math
import torch
from torch.utils.data import DataLoader

# customized modules
from dataTools.factory import *
from models.factory import *
from losses.factory import *
from losses.entropyMinimization import loss_entropy
from utils import fix_random_seed, validate, train
from dataTools.mixupDataset import MixupDataset
from customized.mixup import mixup_two_targets, mixup_bce
from dataTools.utils import *
from os import makedirs
from os.path import join,exists
import settings
# ================

# ## define program arguments
def get_params():
    parser = argparse.ArgumentParser(description='Dist-PU: Positive-Unlabeled Learning From a Label Distribution Perspective')

    parser.add_argument('--device', type=int, default=0, help='GPU index')
    parser.add_argument('--dataset', type=str, default='cifar-10', choices=['TEP','DAMADICS'])
    parser.add_argument('--datapath', type=str, default=settings.TEP_DATA_DIR) # TODO: fill in the datapath
    parser.add_argument('--caseid', type=str, default="",help = 'identifier in folder name for each test run')
    
    parser.add_argument('--TEP-fault-number', type=int, default=1)
    parser.add_argument('--TEP-Positive-Ratio', type=float, default=0.4)
    parser.add_argument('--TEP-Fault-Ratio', type=float, default=0.2)
    parser.add_argument('--num-labeled', type=int, default=int(4000*0.8), help='num of labeled positives in training set')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--loss', type=str, default='Dist-PU', choices=['Dist-PU'], help='PU risk')
    
    parser.add_argument('--warm-up-lr', type=float, default=5e-2, help='Learning rate')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--warm-up-weight-decay', type=float, default=5e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)

    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam'])
    parser.add_argument('--schedular', type=str, default='cos-ann', choices=['cos-ann'])
    parser.add_argument('--entropy', type=int, default=0, choices=[0, 1])

    parser.add_argument('--co-mu', type=float, default=2e-3, help='coefficient of L_ent')
    parser.add_argument('--co-entropy', type=float, default=0.004)

    parser.add_argument('--alpha', type=float, default=6.0)
    parser.add_argument('--co-mix-entropy', type=float, default=0.04)
    parser.add_argument('--co-mixup', type=float, default=5.0)

    parser.add_argument('--warm-up-epochs', type=int, default=40)
    parser.add_argument('--pu-epochs', type=int, default=40)

    parser.add_argument('--random-seed', type=int, default=0,
                        help='initial conditions for generating random variables')

    global args
    args = parser.parse_args()
    print(args)

    return args
# ================


# ## experiment set up
def set_up_for_warm_up():
    fix_random_seed(args.random_seed)

    # set device
    global device
    device = torch.device('cuda:{}'.format(args.device) 
        if torch.cuda.is_available() else "cpu")
    args.device = device

    # obtain data
    global dataset_train, dataset_test, pu_dataset, test_loader
    dataset_train, dataset_test = create_dataset(args.dataset, args.datapath,args)
    args.num_labeled = int((dataset_train.Y).sum())
    pu_dataset = create_pu_dataset(dataset_train, args.num_labeled)
    test_loader = DataLoader(
        dataset_test, batch_size=args.test_batch_size, num_workers=args.num_workers, 
        shuffle=False, pin_memory=True
    )

    global train_loader
    train_loader = DataLoader(
        pu_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
        shuffle=True, pin_memory=True
    )

    # obtain model
    global model
    model = create_model(args.dataset)
    model = model.to(device)
    print (model)

    # loss fn
    global loss_fn
    loss_fn = create_loss(args)

    # obtain optimizer
    global optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.warm_up_lr,
            weight_decay=args.warm_up_weight_decay
        )
    else:
        raise NotImplementedError("The optimizer: {} is not defined!".format(args.optimizer))

    # obtain schedular
    global schedular
    if args.schedular == 'cos-ann':
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.pu_epochs)
    else:
        raise NotImplementedError("The schedular: {} is not defined!".format(args.schedular))
        
    global result_valid
    result_valid = np.zeros((args.warm_up_epochs+args.pu_epochs,len(dataset_test)))
    
    return

def set_up_for_dist_pu():
    # mixup dataset
    global mixup_loader, mixup_dataset
    mixup_loader = DataLoader(
        pu_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, 
        shuffle=False, pin_memory=True
    )

    mixup_dataset = MixupDataset()
    mixup_dataset.update_psudos(mixup_loader, model, device)

    # label distribution loss
    global base_loss
    args.entropy = 0
    base_loss = create_loss(args)

    global co_entropy
    co_entropy = 0

    # obtain optimizer
    global optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError("The optimizer: {} is not defined!".format(args.optimizer))

    # obtain schedular
    global schedular
    if args.schedular == 'cos-ann':
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.pu_epochs, 0.7*args.lr)
    else:
        raise NotImplementedError("The schedular: {} is not defined!".format(args.schedular))
    print("set_up_for_dist_pu")
    return
# ================

# ## train script with mixup
def train_mixup():
    model.train()
    loss_total = 0
    for _, (index, Xs, Ys) in enumerate(train_loader):
        Xs = Xs.to(device)
        Ys = Ys.to(device)
        psudos = mixup_dataset.psudo_labels[index].to(device)
        psudos[Ys==1] = 1

        mixed_x, y_a, y_b, lam = mixup_two_targets(Xs, psudos, args.alpha, device)
        outputs = model(mixed_x).squeeze()
        outputs = torch.clamp(outputs, min=-10, max=10)
        scores = torch.sigmoid(outputs)

        outputs_ = torch.clamp(model(Xs).squeeze(), min=-10, max=10)
        scores_ = torch.sigmoid(outputs_)

        loss = ( base_loss(outputs_, Ys.float())
            + co_entropy*loss_entropy(scores_[Ys!=1]) 
            + args.co_mix_entropy*loss_entropy(scores)
            + args.co_mixup*mixup_bce(scores, y_a, y_b, lam))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mixup_dataset.psudo_labels[index] = scores_.detach()

        loss_total = loss_total + loss.item()

    
    schedular.step()
    return loss_total / len(train_loader)
# ================

# ## functions used in main
def warm_up():
    global result_valid
    for epoch in range(args.warm_up_epochs):
        train_loss = train(train_loader, model, device, loss_fn, optimizer, schedular)
        test_loss, acc, precision, recall, f1, auc, ap, predicted_scores= validate(test_loader, model, device, loss_fn)
        print('epoch:{}; loss: {:.4f}, {:.4f}; \
            acc: {:.5f}; precision: {:.5f}; recall: {:.5f}, f1: {:.5f}, auc: {:.5f}, ap: {:.5f}'.format(
                epoch, train_loss, test_loss, acc, precision, recall, f1, auc, ap
            ))
        result_valid[epoch] = predicted_scores
        
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        plt.figure()
        x=np.linspace(0, len(predicted_scores), len(predicted_scores),False)
        plt.scatter(x,predicted_scores,label="pred",s=1)
        # plt.title(title)
        plt.legend(loc="upper center")
        plt.ylim(0,1)
        plt.savefig(join(settings.FIGURE_DIR,settings.TRAINING_PROCESS_SUBFOLDER,r"%d.png"%(epoch)))
        plt.close()
        
        print ("Warm Up Epoch %d"%(epoch))
    
    print('Final result is %g', acc)
        

def update_co_entropy(epoch):
    global co_entropy
    co_entropy = (1-math.cos((float(epoch)/args.pu_epochs) * (math.pi/2))) * args.co_entropy

def dist_PU():
    global result_valid
    test_loss, acc, precision, recall, f1, auc, ap, predicted_scores = validate(test_loader, model, device, base_loss)
    print('pretrained; loss: {:.4f}; \
        acc: {:.5f}; precision: {:.5f}; recall: {:.5f}, f1: {:.5f}, auc: {:.5f}, ap: {:.5f}'.format(
            test_loss, acc, precision, recall, f1, auc, ap
        ))

    best_acc = 0
    for epoch in range(args.pu_epochs):
        update_co_entropy(epoch)
        print('==> updating co-entropy: {:.5f}'.format(co_entropy))

        print('==> training with mixup')
        train_loss = train_mixup()
        test_loss, acc, precision, recall, f1, auc, ap,predicted_scores = validate(test_loader, model, device, base_loss)
        print('epoch:{}; loss: {:.4f}, {:.4f}; \
            acc: {:.5f}; precision: {:.5f}; recall: {:.5f}, f1: {:.5f}, auc: {:.5f}, ap: {:.5f}'.format(
                epoch, train_loss, test_loss, acc, precision, recall, f1, auc, ap
            ))
    
    
        result_valid[args.warm_up_epochs+epoch] = predicted_scores
        print ("Dist_PU Epoch %d"%(epoch))
        
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        plt.figure()
        x=np.linspace(0, len(predicted_scores), len(predicted_scores),False)
        plt.scatter(x,predicted_scores,label="pred",s=1)
        # plt.title(title)
        plt.ylim(0,1)
        plt.legend(loc="upper center")
        plt.savefig(join(settings.FIGURE_DIR,settings.TRAINING_PROCESS_SUBFOLDER,r"%d.png"%(epoch)))
        plt.close()
        
        if acc > best_acc:
            best_acc = acc
    print('Best result is %g', best_acc)
    print('Final Result is %g', acc)
# ================


def SaveDistPU(model,title = "DistPU"):
    import pickle
    savepath = join(settings.MODEL_DIR,"%s/"%(title))
    if not exists(savepath):
        makedirs(savepath)
    torch.save(model,join(savepath,"DistPU.pth"))
    global result_valid
    with open(join(savepath,"validation.pkl"),"wb") as output:
        pickle.dump(result_valid, output)
    
def LoadDistPU(model,title = "DistPU"):
    import pickle
    import os
    savepath = join(settings.MODEL_DIR,"%s/"%(title))
    if not exists(savepath):
        return 
    model = torch.load(join(savepath,"DistPU.pth"))
    global result_valid
    with open(join(savepath,"validation.pkl"),"rb") as input:
        result_valid = pickle.load(input)

# ## main
if __name__ == '__main__':
    try:
        get_params()
        # args.datapath = "../../public_data/cifar-10-batches-py"
        # args.device = "0"
        # args.dataset = "DAMADICS"

        print('====> warm up')
        set_up_for_warm_up()
        warm_up()

        print('====> Dist-PU')
        set_up_for_dist_pu()
        dist_PU()
        
        if args.dataset == "DAMADICS":
            title = "DistPU(DAM-09-17#3)[F%d%%P%d%%]%s"%(
                args.TEP_Fault_Ratio*100,args.TEP_Positive_Ratio*100,args.caseid)
        else :
            title = "DistPU(TEP-%d)[F%d%%P%d%%]"%(1,args.TEP_Fault_Ratio*100,args.TEP_Positive_Ratio*100)
        print (title)
        SaveDistPU(model,title)
        
        import matplotlib
        matplotlib.use('Qt5Agg')
        Yt = test_loader.dataset.Y
        A,P,R,F = TrainingPlot(result_valid,Yt)
        matplotlib.pyplot.title(title)
        matplotlib.pyplot.savefig(join(settings.FIGURE_DIR,r"%s.png"%(title)))
        
    except Exception as exception:
        print(exception)
        raise
# ================