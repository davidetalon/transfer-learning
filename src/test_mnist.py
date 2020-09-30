import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData, CelebA
from torchvision.models import resnet18, vgg
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from utils.correlation import max_corr, soft_max_corr
from utils.model import initilize_resNet
from model.baseline import ShallowLinear, FilmLayer
import matplotlib.pyplot as plt
from data.dataset import CIFARDataset, MultitaskLabels, MNISTDataset
import pickle
import numpy as np

torch.manual_seed(5)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

pre_train_epochs = 5
mtlcorr_epochs = 0

tensors=[]
def get_intermediate(self, input, output):
    tensors.append(output)
    # tensors.append(output[:, :int(output.shape[1]//2), :, :])

handles = []
def register_hooks(m):
    if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
        handles.append(m.register_forward_hook(get_intermediate))
        m.eval()

def bn_requires_grad(m):
    if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
        # m.eval()
        # m.reset_running_stats()
        # m.track_running_stats = False
        # m.momentum = 0.3
        for param in m.parameters():
            param.requires_grad = True

def evaluate(model, dloader, device):
    model.eval() 
    with torch.no_grad():
        
        correct = 0
        total = 0
        for i, batch in enumerate(dloader):
            sample = batch[0].to(device)
            labels = batch[1].to(device)

            out = model(sample)
            _, pred = torch.max(out, dim=1)

            correct += (labels==pred).float().sum().item()
            total += labels.size(0)
        
    model.train()
    return correct/ total


def train_mtl_corr(batch, models, losses, optimizers, weights):

    w_t1 = weights['w_t1']
    w_t2 = weights['w_t2']
    w_corr = weights['w_corr']
    w_spar = weights['w_spar']
    w_soft = weights['w_soft']

    T = 20

    optimizers[0].zero_grad()
    optimizers[1].zero_grad()

    samples = batch[0].to(device)
    labels = batch[1].to(device)
    soft_labels_t1 = batch[2][0].to(device)
    soft_labels_t2 = batch[2][1].to(device)

    pred = models[0](samples)
    # _, out_pred = torch.max(pred, dim=1)
    t1_loss = losses[0](pred, labels[:, 0])

    soft_t1 = nn.KLDivLoss()(F.log_softmax(pred/T, dim=1), F.softmax(soft_labels_t1/T, dim=1))*T*T

    pred = models[1](samples)
    # _, out_pred = torch.max(pred, dim=1)
    t2_loss = losses[1](pred, labels[:, 1])
    soft_t2 = nn.KLDivLoss()(F.log_softmax(pred/T, dim=1), F.softmax(soft_labels_t2/T, dim=1))*T*T

    shared_corr = 0
    spec_corr = 0
    if w_corr != 0:
        num_intermediate = len(tensors)//2
        for layer in range(num_intermediate):
            shape = tensors[layer].shape
            shared_t1 = tensors[layer][:, :int(shape[1]//2), :, :]
            shared_t2 = tensors[num_intermediate+layer][:, :int(shape[1]//2), :, :]
            shared_corr = shared_corr + max_corr(shared_t1, shared_t2, device)

            spec_t1 = tensors[layer][:, int(shape[1]//2):, :, :]
            spec_t2 = tensors[num_intermediate+layer][:, int(shape[1]//2):, :, :]
            spec_corr = spec_corr + max_corr(spec_t1, spec_t2, device)


    mse = 0
    # for layer in range(20):
    #     dist = torch.nn.functional.pairwise_distance(tensors[layer], tensors[20+layer], p=2.0, eps=1e-06, keepdim=False).sum()
    #     mse = mse + dist 
    #     mse = mse + losses[2](tensors[layer], tensors[20+layer])

    spar_loss = 0
    if w_spar != 0:
        spar_loss = spar(models[0], models[1])

    # loss = w_t1*t1_loss + w_t2*t2_loss
    corr = spec_corr - shared_corr
    loss = w_t1*t1_loss + w_t2*t2_loss + w_corr*corr + w_spar*spar_loss + w_soft*(soft_t1 + soft_t2)
    # loss = t1_loss + t2_loss

    
    loss.backward()
    optimizers[0].step()
    optimizers[1].step()

    return loss, t1_loss, t2_loss, shared_corr, spec_corr, mse, spar_loss, soft_t1, soft_t2

orig_model_t1 = initilize_resNet(out_classes=2, pretrained=None, load='models/res18_STL_t1.ckp', device=device)
orig_model_t1 = dict(orig_model_t1.named_parameters())
orig_model_t2 = initilize_resNet(out_classes=2, pretrained=None, load='models/res18_STL_t2.ckp', device=device)
orig_model_t2 = dict(orig_model_t2.named_parameters())
def spar(model_t1, model_t2):

    spar_t1 = 0
    for name, param in model_t1.named_parameters():
        spar_t1 = spar_t1 + torch.norm((orig_model_t1[name]- param), 2)
        # if (name.find('fc') == -1):
        #     spar_t1 = spar_t1 + torch.norm((orig_model_t1[name]- param), 2)
        # else:
        #     # the last layer is not constrained to previous one?!
        #     spar_t1 = spar_t1 + torch.norm(param, 2)

    spar_t2 = 0
    for name, param in model_t2.named_parameters():
        spar_t2 = spar_t2 + torch.norm((orig_model_t2[name]- param), 2)
        # if (name.find('fc') == -1):
        #     spar_t2 = spar_t2 + torch.norm((orig_model_t2[name]- param), 2)
        # else:
        #     # the last layer is not constrained to previous one?!
        #     spar_t2 = spar_t2 + torch.norm(param, 2)

    return spar_t1 + spar_t2


# transform=ToTensor()
transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

train_t1 = CelebA('data/raw/CelebA', split='train', target_type='attr', transform=None, target_transform=None, download=True)
dloader_train_t1 = torch.utils.data.DataLoader(train_t1, batch_size=32, shuffle=True, num_workers=0)

ft_t1 = CelebA('data/raw/CelebA', split='valid', target_type='attr', transform=None, target_transform=None, download=True)
dloader_ft_t1 = torch.utils.data.DataLoader(ft_t1, batch_size=32, shuffle=True, num_workers=0)

test_t1 = CelebA('data/raw/CelebA', split='test', target_type='attr', transform=None, target_transform=None, download=True)
dloader_test_t1 = torch.utils.data.DataLoader(test_t1, batch_size=32, shuffle=True, num_workers=0)


# train_t1 = CelebA('data/raw/CelebA', split='train', target_type='attr', transform=None, target_transform=None, download=True)
# dloader_train_t1 = torch.utils.data.DataLoader(train_t1, batch_size=32, shuffle=True, num_workers=0)

# train_t1 = MNISTDataset(root='data/processed/MNIST', classes=[6,8], image_set = 'train', transform=transform)
# dloader_train_t1 = torch.utils.data.DataLoader(train_t1, batch_size=32, shuffle=True, num_workers=0)

# ft_t1 = MNISTDataset(root='data/processed/MNIST', classes=[6,8], image_set = 'ft', transform=transform)
# dloader_ft_t1 = torch.utils.data.DataLoader(ft_t1, batch_size=32, shuffle=True, num_workers=0)

# test_t1 = MNISTDataset(root='data/processed/MNIST', classes=[6,8], image_set='test', transform=transform)
# dloader_test_t1 = torch.utils.data.DataLoader(test_t1, batch_size=32, shuffle=False, num_workers=0)

###############################################

train_t2 = MNISTDataset(root='data/processed/MNIST', classes=[1,7], image_set = 'train', transform=transform)
dloader_train_t2 = torch.utils.data.DataLoader(train_t2, batch_size=32, shuffle=True, num_workers=0)

ft_t2 = MNISTDataset(root='data/processed/MNIST', classes=[1,7], image_set = 'ft', transform=transform)
dloader_ft_t2 = torch.utils.data.DataLoader(ft_t2, batch_size=32, shuffle=True, num_workers=0)

test_t2 = MNISTDataset(root='data/processed/MNIST', classes=[1,7], image_set='test', transform=transform)
dloader_test_t2 = torch.utils.data.DataLoader(test_t2, batch_size=32, shuffle=False, num_workers=0)

######################################################


models=[]
models.append(initilize_resNet(out_classes=2, pretrained=False, load=None, device=device))
models[0].conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
models.append(initilize_resNet(out_classes=2, pretrained=False, load=None, device=device))
models[1].conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)

t1_criterion = torch.nn.CrossEntropyLoss().to(device)
t2_criterion = torch.nn.CrossEntropyLoss().to(device)


opt_t1 = torch.optim.SGD(models[0].parameters(),lr=0.01, momentum=0.9, weight_decay=5e-4)
opt_t2 = torch.optim.SGD(models[1].parameters(),lr=0.01, momentum=0.9, weight_decay=5e-4)


#####################################
# TRAIN T1
#####################################

pre_test_acc_t1 = []
pre_test_acc_t1.append(evaluate(models[0], dloader_test_t1, device=device))

pre_train_acc_t1 = []
pre_train_acc_t1.append(evaluate(models[0], dloader_train_t1, device=device))

pre_ft_acc_t1 = []
pre_ft_acc_t1.append(evaluate(models[0], dloader_ft_t1, device=device))

for epoch in range(pre_train_epochs):
    for i, batch in enumerate(dloader_train_t1):

        opt_t1.zero_grad()

        sample = batch[0].to(device)
        labels = batch[1].to(device)
        pred = models[0](sample)

        t1_loss = t1_criterion(pred, labels)
        t1_loss.backward()
        opt_t1.step()


        print('[T1 PRE-TRAIN][Epoch %d/%d][Batch %d/%d][Loss %f]' % (epoch, pre_train_epochs, i, len(dloader_train_t1), t1_loss))
    
    pre_test_acc_t1.append(evaluate(models[0], dloader_test_t1, device=device))
    pre_train_acc_t1.append(evaluate(models[0], dloader_train_t1, device=device))
    pre_ft_acc_t1.append(evaluate(models[0], dloader_ft_t1, device=device))


#####################################
# TRAIN T2
#####################################

pre_test_acc_t2 = []
pre_test_acc_t2.append(evaluate(models[1], dloader_test_t2, device=device))

pre_train_acc_t2 = []
pre_train_acc_t2.append(evaluate(models[1], dloader_train_t2, device=device))

pre_ft_acc_t2 = []
pre_ft_acc_t2.append(evaluate(models[1], dloader_ft_t2, device=device))

for epoch in range(pre_train_epochs):
    for i, batch in enumerate(dloader_train_t2):

        opt_t2.zero_grad()

        sample = batch[0].to(device)
        labels = batch[1].to(device)

        pred = models[1](sample)
        t2_loss = t2_criterion(pred, labels)
        t2_loss.backward()
        opt_t2.step()

        print('[T2 PRE-TRAIN][Epoch %d/%d][Batch %d/%d][Loss %f]' % (epoch, pre_train_epochs, i, len(dloader_train_t2), t2_loss))
    
    pre_test_acc_t2.append(evaluate(models[1], dloader_test_t2, device=device))
    pre_train_acc_t2.append(evaluate(models[1], dloader_train_t2, device=device))
    pre_ft_acc_t2.append(evaluate(models[1], dloader_ft_t2, device=device))

#################################################################



raise SystemError(0)


# print("Saving t1")
# torch.save(models[0].state_dict(), "runs_res18_STL_t1.ckp")

# print("Saving t2")
# torch.save(models[1].state_dict(), "runs_res18_STL_t2.ckp")



runs = [
        {'w_t1': 1, 'w_t2': 1, 'w_corr': 0.1, 'w_spar': 0, 'w_soft': 0.1, 'bn_ft': True, 'lr': 0.001, 'decay': False},
        # {'w_t1': 1, 'w_t2': 1, 'w_corr': 0, 'w_spar': 0, 'w_soft': 0.1, 'bn_ft': True, 'lr': 0.001, 'decay': False},
        # {'w_t1': 1, 'w_t2': 1, 'w_corr': 0.1, 'w_spar': 0, 'w_soft': 0, 'bn_ft': True, 'lr': 0.001, 'decay': False},
        # {'w_t1': 1, 'w_t2': 1, 'w_corr': 0, 'w_spar': 0, 'w_soft': 0, 'bn_ft': True, 'lr': 0.001, 'decay': False},
        
]       

# w_t = [0.1, 0.2, 0.5, 1]
# w_corr = [0.1, 0.2, 0.5, 1]
# w_soft = [0.1, 0.2]

# runs=[]
# for wt in w_t:
#     for wcorr in w_corr:
#         for wsoft in w_soft:
#             runs.append({'w_t1':wt, 'w_t2':wt, 'w_corr':wcorr, 'w_spar':0, 'w_soft':wsoft, 'bn_ft':True, 'lr':0.001, 'decay':False})

# n_runs = 5
# runs=[]
# for no_corr_idx in range(n_runs):
#     runs.append({'w_t1': 1, 'w_t2': 1, 'w_corr': 0.2, 'w_spar': 0, 'w_soft': 0.1, 'bn_ft': True, 'lr': 0.001, 'decay': False})
# for corr_idx in range(n_runs):
#     runs.append({'w_t1': 1, 'w_t2': 1, 'w_corr': 0, 'w_spar': 0, 'w_soft': 0.1, 'bn_ft': True, 'lr': 0.001, 'decay': False})
# for corr_idx in range(n_runs):
#     runs.append({'w_t1': 1, 'w_t2': 1, 'w_corr': 0.2, 'w_spar': 0, 'w_soft': 0, 'bn_ft': True, 'lr': 0.001, 'decay': False})
   

run_test_acc_t1 = []
run_test_acc_t2 = []

run_train_acc_t1=[]
run_train_acc_t2=[]

run_ft_acc_t1=[]
run_ft_acc_t2=[]

for run_idx, run in enumerate(runs):
    
    models=[]
    models.append(initilize_resNet(out_classes=2, pretrained=None, load='models/res18_STL_t1.ckp', device=device))
    models.append(initilize_resNet(out_classes=2, pretrained=None, load='models/res18_STL_t2.ckp', device=device))

    if ft_mtask.requires_soft==False:
        ft_mtask.produce_soft_labels(models, 128, device)
    

    opt_t1 = torch.optim.SGD(models[0].parameters(),lr=run['lr'], momentum=0.9)
    opt_t2 = torch.optim.SGD(models[1].parameters(),lr=run['lr'], momentum=0.9)

    if run['decay']:
        sched_t1 = torch.optim.lr_scheduler.StepLR(opt_t1, step_size=2, gamma=0.75)
        sched_t2 = torch.optim.lr_scheduler.StepLR(opt_t2, step_size=2, gamma=0.75)

    # test_acc_t1 = []
    # test_acc_t1.extend(pre_test_acc_t1)
    # test_acc_t2 = []
    # test_acc_t2.extend(pre_test_acc_t2)

    # train_acc_t1 = []
    # train_acc_t1.extend(pre_train_acc_t1)
    # train_acc_t2 = []
    # train_acc_t2.extend(pre_train_acc_t2)

    # ft_acc_t1 = []
    # ft_acc_t1.extend(pre_ft_acc_t1)
    # ft_acc_t2 = []
    # ft_acc_t2.extend(pre_ft_acc_t2)

    test_acc_t1 = []
    test_acc_t1.append(evaluate(models[0], dloader_test_t1, device=device))
    test_acc_t2 = []
    test_acc_t2.append(evaluate(models[1], dloader_test_t2, device=device))

    train_acc_t1 = []
    train_acc_t1.append(evaluate(models[0], dloader_train_t1, device=device))
    train_acc_t2 = []
    train_acc_t2.append(evaluate(models[1], dloader_train_t2, device=device))

    ft_acc_t1 = []
    ft_acc_t1.append(evaluate(models[0], dloader_ft_t1, device=device))
    ft_acc_t2 = []
    ft_acc_t2.append(evaluate(models[1], dloader_ft_t2, device=device))

    
    if run['bn_ft']:
        for model in models:
            for param in model.parameters():
                param.requires_grad = False
            model.apply(bn_requires_grad)
            for param in model.fc.parameters():
                param.requires_grad = True

    for epoch in range (mtlcorr_epochs):
        for i, batch in enumerate(dloader_ft_mtask):

            models[0].apply(register_hooks)
            models[1].apply(register_hooks)

            loss, t1_loss, t2_loss, shared_corr, spec_corr, mse, spar_loss, soft_t1, soft_t2 = train_mtl_corr(batch, 
                                                                                                            models=models,
                                                                                                            losses=[t1_criterion, t2_criterion], 
                                                                                                            optimizers=[opt_t1, opt_t2], 
                                                                                                            weights=run)
            # loss = t1_loss + t2_loss - corr
            tensors = []

            print('[MTL-corr][Run %d/%d][Epoch %d/%d][Batch %d/%d][Loss %f][Corr %f/%f][MSE %f][Spar %f][Soft %f\%f][Class %f\%f]' % 
            (run_idx, len(runs), epoch, mtlcorr_epochs, i, len(dloader_ft_mtask), loss, shared_corr, spec_corr, mse, spar_loss, soft_t1, soft_t2, t1_loss, t2_loss))


            for handle in handles:
                handle.remove()

        if run['decay']:
            sched_t1.step()
            sched_t2.step()

        test_acc_t1.append(evaluate(models[0], dloader_test_t1, device=device))
        test_acc_t2.append(evaluate(models[1], dloader_test_t2, device=device))

        train_acc_t1.append(evaluate(models[0], dloader_train_t1, device=device))
        train_acc_t2.append(evaluate(models[1], dloader_train_t2, device=device))

        ft_acc_t1.append(evaluate(models[0], dloader_ft_t1, device=device))
        ft_acc_t2.append(evaluate(models[1], dloader_ft_t2, device=device))

    run_test_acc_t1.append(test_acc_t1)
    run_test_acc_t2.append(test_acc_t2)
    run_train_acc_t1.append(train_acc_t1)
    run_train_acc_t2.append(train_acc_t2)
    run_ft_acc_t1.append(ft_acc_t1)
    run_ft_acc_t2.append(ft_acc_t2)

data = {'test_t1':run_test_acc_t1, 
        'test_t2':run_test_acc_t2,
        'train_t1':run_train_acc_t1,
        'train_t2':run_train_acc_t2,
        'ft_t1':run_ft_acc_t1,
        'ft_t2':run_ft_acc_t2,
        'runs':runs}

with open('bn_metrics.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


gammas = []
betas = []
for m in models[0].modules():
    if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
        for name,param in m.named_parameters():
            if name == 'weight':
                gammas.append(param.detach().cpu().numpy().flatten())
            else:
                betas.append(param.detach().cpu().numpy().flatten())

gammas = np.concatenate(gammas, axis=0)
betas = np.concatenate(betas, axis = 0)
plt.hist(gammas, bins=2000, density=True)
plt.savefig('bn_gammas_t1.png')
plt.close()

plt.hist(betas, bins=2000, density=True)
plt.savefig('bn_betas_t1.png')
plt.close()

gammas = []
betas = []
for m in models[1].modules():
    if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
        for name,param in m.named_parameters():
            if name == 'weight':
                gammas.append(param.detach().cpu().numpy().flatten())
            else:
                betas.append(param.detach().cpu().numpy().flatten())

gammas = np.concatenate(gammas, axis=0)
betas = np.concatenate(betas, axis = 0)
plt.hist(gammas, bins=200, density=True)
plt.savefig('bn_gammas_t2.png')
plt.close()

plt.hist(betas, bins=200, density=True)
plt.savefig('bn_betas_t2.png')
plt.close()



for i, run in enumerate(runs):
    plt.plot(run_test_acc_t1[i], label=str(i))
plt.legend(loc='lower left')
plt.savefig('bn_t1.png')
plt.close()

for i, run in enumerate(runs):
    plt.plot(run_test_acc_t2[i], label=str(i))

plt.legend(loc='lower left')
plt.savefig('bn_t2.png')
plt.close()

# plotting learning curves
plt.plot(run_test_acc_t1[0], label='test')
plt.plot(run_train_acc_t1[0], label='train')
plt.plot(run_ft_acc_t1[0], label='ft')
plt.legend(loc='lower left')
plt.savefig('bn_lc_t1.png')
plt.close()

plt.plot(run_test_acc_t2[0], label='test')
plt.plot(run_train_acc_t2[0], label='train')
plt.plot(run_ft_acc_t2[0], label='ft')
plt.legend(loc='lower left')
plt.savefig('bn_lc_t2.png')
plt.close()

