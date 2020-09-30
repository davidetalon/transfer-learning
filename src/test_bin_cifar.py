import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.models import resnet50, vgg, resnet18
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.transforms import ToTensor, Compose, Normalize, RandomResizedCrop, CenterCrop
from utils.correlation import max_corr, soft_max_corr
from utils.model import initilize_resNet, Intermediate
from model.baseline import ShallowLinear, FilmLayer,soft_hist, GaussianHistogram, SoftHist
import matplotlib.pyplot as plt
from data.dataset import CIFARDataset, MultitaskLabels, BinaryCIFAR, Places365, imshow
import pickle
import numpy as np
import random
import seaborn as sns
import math
import datetime
import time
from pathlib import Path

from PIL import Image

torch.manual_seed(5)
random.seed(5)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

metrics_folder = 'saved_metrics'
metrics_folder = Path(metrics_folder)
metrics_folder.mkdir(parents=True, exist_ok=True)

date = datetime.datetime.now()
date = date.strftime("%d-%m-%Y_%H-%M-%S")
out_dir = metrics_folder / str(date)
out_dir.mkdir(parents=True, exist_ok=True)


pre_train_epochs = 0
mtlcorr_epochs = 150
n_tasks = 2

def bn_requires_grad(m):

    if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
        m.reset_parameters()
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


def train_mtl_corr(batches, models, criterions, optims, weights, hooks_manager):

    loss_weights = weights['w_t']
    w_corr_sh = weights['w_corr_sh']
    w_corr_spec = weights['w_corr_spec']
    w_spar = weights['w_spar']
    w_recon = weights['w_recon']
    soft_loss_weights = weights['w_soft']

    
    T = 20

    losses = []
    soft_losses = []
    for task, model in enumerate(models):
        
        optims[task].zero_grad()
        fc_optims[task].zero_grad()

        # load data
        samples = batches[task][0].to(device)
        labels = batches[task][1].to(device)
        soft_labels = batches[task][2].to(device)

        pred = model(samples)
        losses.append(criterions[task](pred, labels))
        soft_losses.append(nn.KLDivLoss()(F.log_softmax(pred/T, dim=1), F.softmax(soft_labels/T, dim=1))*T*T)
    
    shared_corr = 0
    max_corr_1 = 0
    max_corr_2 = 0
    gather_correlations = []
    reconstruction_loss = 0

    n_tasks = len(models)
    if w_corr_sh != 0:
        n_layers = len(hooks_manager[0].tensors)
        count_combinations = 0
        
        for s in range(n_tasks):
            for t in range(s+1, n_tasks):

                if s==t:
                    continue

                count_combinations = count_combinations + 1
                source = hooks_manager[s].tensors
                target = hooks_manager[t].tensors

                # for each layer compute the correlation between shared subspaces
                for layer in range(n_layers):

                    # layer_corr_w = torch.arange(n_layers, device=device, dtype=torch.float).flip(dims=[0])
                    # layer_corr_w = torch.add(layer_corr_w, 1)
                    layer_corr_w = torch.ones(n_layers, device=device)
    

                    shape = hooks_manager[s].tensors[layer].shape
                    shared_s = source[layer][:, :int(shape[1]//2), :, :]
                    shared_t = target[layer][:, :int(shape[1]//2), :, :]
                    # shared_corr = shared_corr + max_corr(shared_s, shared_t, device)
                    # print('soft_corr', soft_max_corr(shared_s.detach(), shared_t.detach(), device))
                    # print('max_corr', max_corr(shared_s.detach(), shared_t.detach(), device))
                    # print(shared_s.shape)

                    ###################################
                    # #global corr
                    # h_corr = max_corr(shared_s, shared_t, device)
                    # shared_corr = shared_corr + h_corr
                    # corr = h_corr.item()


                    ###################################
                    # histogram

                    # hist_s = torch.histc(shared_s[0].reshape(-1), bins=16, min=0, max=5)

                    # src_hists = []
                    # tr_hists = []

                    # # kl_div = 0
                    # for idx, sample in enumerate(torch.split(shared_s, 1)):
                    #     shist = soft_hist(sample.reshape(-1), bins=16, min=0, max=5, sigma=100)

                    #     src_hists.append(shist.unsqueeze(0))
                    #     thist = soft_hist(shared_t[idx].reshape(-1), bins=16, min=0, max=5, sigma=100)
                    #     tr_hists.append(thist.unsqueeze(0))
        
                    #     # kl_div = kl_div + F.kl_div(F.log_softmax(shist, dim=0), F.softmax(thist, dim=0))
                    # shist = torch.cat(src_hists, dim=0)
                    # thist = torch.cat(tr_hists, dim=0)

                    # kl_div = F.kl_div(F.log_softmax(shist, dim=1), F.softmax(thist, dim=1))
                    # shared_corr = shared_corr - kl_div

                    # # shared_corr = shared_corr + max_corr(shist, thist, device=device)
                    # corr = kl_div.item()
                    ####################################
                    # # stateful hist

                    # s_hists = hist_layers[0][layer](shared_s)
                    # t_hists = hist_layers[1][layer](shared_t)
                    # reconstruction_loss = 0

                    # height = shared_s.shape[2]
                    # width = shared_s.shape[3]
                    # n_channels = shared_s.shape[1]
                    # kls = torch.empty((n_channels, height, width), device=device)

                    # for ch in range(n_channels):
                    #     for i in range(height):
                    #         for j in range(width):

                    #             kls[ch, i, j] = F.kl_div(t_hists[ch, i, j].log(), s_hists[ch, i, j])

                    # corr = kls.mean()
                    # shared_corr = shared_corr - corr
                    # corr = corr.item()

                    #####################################
                    # cross batch hist
                    # encoded_s = shared_s
                    # encoded_t = shared_t
                    # reconstruction_loss=0

                    # if w_recon > 0:
                    #     encoded_s = task1_enc[layer](shared_s)
                    #     encoded_t = task2_enc[layer](shared_t)

                    #     shared_s_hat = task1_dec[layer](encoded_s)
                    #     shared_t_hat = task2_dec[layer](encoded_t)
                    #     reconstruction_loss = torch.nn.functional.mse_loss(shared_s, shared_s_hat) + torch.nn.functional.mse_loss(shared_t, shared_t_hat)

                    # n_channels = encoded_s.shape[1]
                    # height = encoded_s.shape[2]
                    # width = encoded_s.shape[3]
                    # n_bins = 8
                    # epsilon = 0.2
                    # kls = torch.empty((n_channels, height, width), device=device)
                    # for ch in range(n_channels):
                    #     for i in range(height):
                    #         for j in range(width):
                                
                    #             batch_feat_s = encoded_s[:, ch, i, j]
                    #             val_min, val_max = batch_feat_s.min().item()-epsilon, batch_feat_s.max().item()+epsilon
                    #             hist_s = soft_hist(batch_feat_s.reshape(-1), bins = n_bins, min=val_min, max=val_max, sigma=100)      
                    #             hist_s = torch.add(hist_s, 1.)
                    #             hist_s = torch.true_divide(hist_s, hist_s.sum())

                    #             batch_feat_t = encoded_t[:, ch, i, j]
                    #             hist_t = soft_hist(batch_feat_t.reshape(-1), bins = n_bins, min=val_min, max=val_max, sigma=100)
                    #             hist_t = torch.add(hist_t, 1.)
                    #             hist_t = torch.true_divide(hist_t, hist_t.sum())

                    #             kls[ch, i, j] = F.kl_div(hist_t.log(), hist_s)

                    
                    # corr = kls.mean()
                    # shared_corr = shared_corr - corr
                    # corr = corr.item()

                    ########################################
                    # separated training
          
                    # max_corr_1 = max_corr_1 + max_corr(shared_s, shared_t.detach(), device)
                    # max_corr_2 = max_corr_2 + max_corr(shared_s.detach(), shared_t, device)
                    # corr = max_corr(shared_s, shared_t, device).item()
                    # shared_corr = shared_corr + corr

                    ########################################
                    # simple feature matching

                    # fm = torch.norm(shared_s - shared_t, 2)
                    # shared_corr = shared_corr - fm / shared_s.shape[0]
                    # corr = fm.item() /shared_s.shape[0]
                    # #######################################
                    # feature encoder

                    encoded_s = shared_s
                    encoded_t = shared_t

                    if w_recon > 0:
                        encoded_s = task1_enc[layer](shared_s)
                        encoded_t = task2_enc[layer](shared_t)

                        shared_s_hat = task1_dec[layer](encoded_s)
                        shared_t_hat = task2_dec[layer](encoded_t)
                        reconstruction_loss = torch.nn.functional.mse_loss(shared_s.detach(), shared_s_hat) + torch.nn.functional.mse_loss(shared_t.detach(), shared_t_hat)
                    
                    n_channels = encoded_s.shape[1]
                    height = encoded_s.shape[2]
                    width = encoded_s.shape[3]
                    kls = torch.empty((n_channels), device=device)
                    for ch in range(n_channels):
                        
                                feat_s = encoded_s[:, ch, :, :]
                                feat_t = encoded_t[:, ch, :, :]

                                kls[ch] = max_corr(feat_s, feat_t, device=device)

                    layer_corr = kls.mean()
                    shared_corr = shared_corr + layer_corr_w[layer] *layer_corr
                    
                    corr = layer_corr.item()

                    ###############################################


                    gather_correlations.append(corr)
     

        shared_corr = shared_corr /n_layers
        max_corr_1 = max_corr_1 /n_layers
        max_corr_2 = max_corr_2 /n_layers




    # for each task at each layer compute the correlation between specidic subspaces
    spec_corr = 0
    if w_corr_spec != 0:            
        for task in range(n_tasks):
            for layer in range(n_layers):
                shape =  hooks_manager[task].tensors[layer].shape
                sh = hooks_manager[task].tensors[layer][:, :int(shape[1]//2), :, :]
                spec = hooks_manager[task].tensors[layer][:, int(shape[1]//2):, :, :]
                spec_corr = spec_corr + max_corr(sh, spec, device)
        spec_corr = spec_corr / (n_tasks * n_layers)



    spar_loss = 0
    if w_spar != 0:
        spar_loss = spar(models)

    corr = w_corr_spec*spec_corr - w_corr_sh*shared_corr

    w_tot_loss =0
    w_tot_soft_loss = 0
    for task in range(len(losses)):
        w_tot_loss = w_tot_loss +  loss_weights[task]*losses[task]
        w_tot_soft_loss = w_tot_soft_loss + soft_loss_weights[task]*soft_losses[task]
    
    # print('losses', losses[0].item(), losses[1].item())
    # print('soft_losses', soft_losses[0].item(), soft_losses[1].item())
    loss = w_tot_loss + corr + w_spar*spar_loss + w_tot_soft_loss

    for task in range(n_tasks):
        autoenc_optims[task].zero_grad()

    loss = w_tot_loss + corr + w_spar*spar_loss + w_tot_soft_loss + w_recon*reconstruction_loss
    loss.backward()
    for task in range(n_tasks):
        optims[task].step()
        fc_optims[task].step()
        autoenc_optims[task].step()
    

    # if w_recon > 0 and w_corr_sh>0:

    #     loss.backward(retain_graph=True)
 
    #     for task in range(n_tasks):
    #         autoenc_optims[task].zero_grad()

    #     recon = w_recon*reconstruction_loss
    #     recon.backward()

    #     for task in range(n_tasks):
    #         optims[task].step()
    #         fc_optims[task].step()
    #         autoenc_optims[task].step()
    # else:

    #     loss.backward()
    #     for task in range(n_tasks):
    #         optims[task].step()
    #         fc_optims[task].step()

        
        

        
    # else:
    #     loss.backward()

    # for task in range(n_tasks):
    #     optims[task].step()
    #     fc_optims[task].step()

    # if w_recon > 0:
    #     for opt in autoenc_optims:
    #         opt.step()

    # for param in task1_enc[2].parameters():
    #     print(param.grad.mean())

    ###################
    #   single task update
    ###################

    # for task in range(n_tasks):
    #     corr = w_corr_sh * shared_corr[task] + w_corr_spec * spec_corr[task]
    #     loss = losses[task] + soft_losses[task]*soft_loss_weights[task]
    #     optims[task].zero_grad()
    #     loss.backward()
    #     optims[task].step()


    # l1 = losses[0] - w_corr_sh * max_corr_1 + soft_losses[0]*soft_loss_weights[0]
    # optims[0].zero_grad()
    # l1.backward()
    # optims[0].step()

    # l1 = losses[1] - w_corr_sh * max_corr_2 + soft_losses[1]*soft_loss_weights[1]
    # optims[1].zero_grad()
    # l1.backward()
    # optims[1].step()

    
    return loss, losses, shared_corr, spec_corr, spar_loss, soft_losses, gather_correlations, reconstruction_loss

orig_models=[]
for task in range(n_tasks):
    orig_model = initilize_resNet(out_classes=5, model='resnet18', pretrained='models/bincifar_res18_5way_t'+str(task)+'.ckp', load=None, device=device)
    orig_models.append(dict(orig_model.named_parameters()))

def spar(models):

    tot_spar = 0
    for task, model in enumerate(models):
        if not (isinstance(model, nn.BatchNorm2d) or isinstance(model, nn.Linear)):
            for name, param in model.named_parameters():
                tot_spar = tot_spar + torch.norm((orig_models[task][name]- param), 2)
    return tot_spar/len(models)



# transform=ToTensor()
transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


dload_train = []
dload_ft = []
dload_val = []
dload_test = []

chosen_tasks = []
for task in range(n_tasks):
    
    task = random.choices(range(100), k=5)
    # task = (1, 91)
    chosen_tasks.append(task)

    train_data = BinaryCIFAR(root='data/processed/cifar/train', tasks=[task], transform=transform)
    dload_train.append(DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4))

    ft_data = BinaryCIFAR(root='data/processed/cifar/fine-tune', tasks=[task], transform=transform)
    dload_ft.append(DataLoader(ft_data, batch_size=64, shuffle=False, num_workers=4, drop_last=False))
    
    val_data = BinaryCIFAR(root='data/processed/cifar/valid', tasks=[task], transform=transform)
    dload_val.append(DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4))

    test_data = BinaryCIFAR(root='data/processed/cifar/test', tasks=[task], transform=transform)
    dload_test.append(DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4))


coarse_label = [
'apple', # id 0
'aquarium_fish',
'baby',
'bear',
'beaver',
'bed',
'bee',
'beetle',
'bicycle',
'bottle',
'bowl',
'boy',
'bridge',
'bus',
'butterfly',
'camel',
'can',
'castle',
'caterpillar',
'cattle',
'chair',
'chimpanzee',
'clock',
'cloud',
'cockroach',
'couch',
'crab',
'crocodile',
'cup',
'dinosaur',
'dolphin',
'elephant',
'flatfish',
'forest',
'fox',
'girl',
'hamster',
'house',
'kangaroo',
'computer_keyboard',
'lamp',
'lawn_mower',
'leopard',
'lion',
'lizard',
'lobster',
'man',
'maple_tree',
'motorcycle',
'mountain',
'mouse',
'mushroom',
'oak_tree',
'orange',
'orchid',
'otter',
'palm_tree',
'pear',
'pickup_truck',
'pine_tree',
'plain',
'plate',
'poppy',
'porcupine',
'possum',
'rabbit',
'raccoon',
'ray',
'road',
'rocket',
'rose',
'sea',
'seal',
'shark',
'shrew',
'skunk',
'skyscraper',
'snail',
'snake',
'spider',
'squirrel',
'streetcar',
'sunflower',
'sweet_pepper',
'table',
'tank',
'telephone',
'television',
'tiger',
'tractor',
'train',
'trout',
'tulip',
'turtle',
'wardrobe',
'whale',
'willow_tree',
'wolf',
'woman',
'worm',
]
for cl in chosen_tasks[1]:
    print(coarse_label[cl])

mtl_ft_data = BinaryCIFAR(root='data/processed/cifar/fine-tune', tasks=chosen_tasks, transform=transform)
dload_mtl_ft = DataLoader(mtl_ft_data, batch_size=64, shuffle=True, num_workers=4, drop_last=True)


models=[]
optims=[]
scheds=[]
criterions = []

for task in range(n_tasks):

    models.append(initilize_resNet(out_classes=5, model='resnet18', pretrained=False, load=None, device=device))
    optims.append(torch.optim.SGD(models[-1].parameters(),lr=0.001, momentum=0.9, weight_decay=5e-4))
    # scheds.append(torch.optim.lr_scheduler.MultiStepLR(optims[-1], [60, 120], gamma=0.2))

    criterions.append(torch.nn.CrossEntropyLoss().to(device))    


train_metrics = []
for task in range(n_tasks):

    train_acc = []
    ft_acc = []
    val_acc = []
    test_acc = []

    for epoch in range(pre_train_epochs):
        for i, batch in enumerate(dload_train[task]):

            optims[task].zero_grad()
            
            sample = batch[0].to(device)
            labels = batch[1].to(device)
            # print(sample)
            pred = models[task](sample)
            loss = criterions[task](pred, labels)

            loss.backward()
            optims[task].step()

            print('[PRE-TRAIN][TASK %d/%d][Epoch %d/%d][Batch %d/%d][Loss %f]' % (task, n_tasks, epoch, pre_train_epochs, i, len(dload_train[task]), loss))


        train_acc.append(evaluate(models[task], dload_train[task], device=device))
        ft_acc.append(evaluate(models[task], dload_ft[task], device=device))
        val_acc.append(evaluate(models[task], dload_val[task], device=device))
        test_acc.append(evaluate(models[task], dload_test[task], device=device))
        
        
        # scheds[task].step()

    train_metrics.append([train_acc, ft_acc, val_acc, test_acc])


train_metrics = np.array(train_metrics)
print(train_metrics.shape) # (n_tasks, n_metrics, n_epochs)


data = {'train_metrics':train_metrics}

# with open(out_dir/('bincifar_train_metrics.pkl'), 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# for task in range(n_tasks):
#     plt.plot(train_metrics[task, -2, :], label='val')
#     plt.plot(train_metrics[task, 0, :], label='train')
#     plt.legend(loc='lower left')
#     plt.savefig(out_dir/('bincifar_5way_training_t'+str(task)+'.png'))
#     plt.close()

# print('Saving models')
# for task, model in enumerate(models):
#     torch.save(model.state_dict(), ('models/bincifar_res18_5way_t'+str(task)+'.ckp'))

# runs = []
# corr = [0.5, 1, 5, 10, 20, 40]
# recon = [0.5, 1, 2]
# lrs = [0.001, 0.005, 0.01]
# for wc in corr:
#     for rec in recon:
#         for lr in lrs:
#             runs.append({'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':4,'w_recon': rec, 'w_corr_sh': wc, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': lr, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},)
runs = [

        # {'w_t':[0.5, 0.5, 0, 0], 'w_corr_sh': 1, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'bn_ft': True, 'lr': 0.01, 'decay': True, 'freeze_stats':False, 'tune-fc':False}, #good validation with convcompr
        # {'w_t':[0.5, 0.5, 0, 0], 'target_ch':32,'w_recon': 1, 'w_corr_sh': 10, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False}, #buoni risultati con batchhist
        # {'w_t':[0.5, 0.5, 0, 0], 'target_ch':1, 'w_recon': 1, 'w_corr_sh': 0.5, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False}, #buono
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':4, 'w_recon': 0, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':4, 'w_recon': 1, 'w_corr_sh': 0.05, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':1, 'w_recon': 0, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},
        
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':4,'w_recon': 5, 'w_corr_sh': 12, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},
       
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':8,'w_recon': 5, 'w_corr_sh': 0.05, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.1, 'decay': True, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},
       
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 12, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 0, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 0.1, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},
        # {'spatial_pooling':True, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},
       
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 0, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':True, 'all-net':True},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 0, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':True, 'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 0, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':False, 'tune-bn':True,'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 0, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':False,'all-net':False},
        
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 0.5, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':True, 'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 1, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':False, 'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 2, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':True, 'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 3, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':True, 'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 4, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':True, 'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':True, 'all-net':True},
        {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 1, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':True, 'all-net':True},

        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 6, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':True, 'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 7, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':True, 'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':True, 'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':True, 'all-net':True},

        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 0, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':False, 'tune-bn':True,'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 0, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.005, 'decay': False, 'freeze_stats':False, 'tune-fc':False, 'tune-bn':True,'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 0, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':False, 'tune-bn':True,'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 0, 'w_corr_sh': 0, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.02, 'decay': False, 'freeze_stats':False, 'tune-fc':False, 'tune-bn':True,'all-net':False},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 1, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'tune-bn':True,'all-net':True},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 0.5, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 1, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},  #buonaaaaaaa
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 2, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 5, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},
        # {'spatial_pooling':False, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 10, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},


        
        # {'spatial_pooling':True, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 20, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},

        # {'spatial_pooling':True, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 0.2, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},
        # {'spatial_pooling':True, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 0.5, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},
        # {'spatial_pooling':True, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 1, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},
        # {'spatial_pooling':True, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 2, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},
        # {'spatial_pooling':True, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 5, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},
        # {'spatial_pooling':True, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0],'target_ch':8,'w_recon': 5, 'w_corr_sh': 10, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},





        # {'spatial_pooling':True, 'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'n_layers': 0, 'target_ch':8,'w_recon': 5, 'w_corr_sh': 0.1, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': True, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'n_layers': 4, 'target_ch':8,'w_recon': 5, 'w_corr_sh': 0.1, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'n_layers': 5, 'target_ch':8,'w_recon': 5, 'w_corr_sh': 0.1, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'n_layers': 6, 'target_ch':8,'w_recon': 5, 'w_corr_sh': 0.1, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'n_layers': 7, 'target_ch':8,'w_recon': 5, 'w_corr_sh': 0.1, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},
        

        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':8,'w_recon': 5, 'w_corr_sh': 0.5, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.1, 'decay': True, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':8,'w_recon': 5, 'w_corr_sh': 1, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.1, 'decay': True, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':8,'w_recon': 5, 'w_corr_sh': 2, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.1, 'decay': True, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':8,'w_recon': 5, 'w_corr_sh': 5, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.1, 'decay': True, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':8,'w_recon': 5, 'w_corr_sh': 10, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.1, 'decay': True, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':8,'w_recon': 5, 'w_corr_sh': 20, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.1, 'decay': True, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},
        
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':8,'w_recon': 2, 'w_corr_sh': 15, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.1, 'decay': True, 'freeze_stats':False, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':8,'w_recon': 0, 'w_corr_sh': 15, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True, 'all-net':True},

        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':8,'w_recon': 5, 'w_corr_sh': 20, 'w_corr_spec': 10, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':16,'w_recon': 5, 'w_corr_sh': 20, 'w_corr_spec': 10, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':32,'w_recon': 5, 'w_corr_sh': 20, 'w_corr_spec': 10, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},

# {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':4,'w_recon': 5, 'w_corr_sh': 20, 'w_corr_spec': 2, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':4,'w_recon': 5, 'w_corr_sh': 20, 'w_corr_spec': 5, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':4,'w_recon': 5, 'w_corr_sh': 20, 'w_corr_spec': 10, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':4,'w_recon': 5, 'w_corr_sh': 20, 'w_corr_spec': 20, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':4,'w_recon': 5, 'w_corr_sh': 20, 'w_corr_spec': 40, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},

        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':4,'w_recon': 5, 'w_corr_sh': 40, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.001, 'decay': False, 'freeze_stats':True, 'tune-fc':True, 'all-net':False},

        # {'batch_size':64, 'w_t':[0.5, 0.5, 0, 0], 'target_ch':4,'w_recon': 5, 'w_corr_sh': 12, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': True, 'freeze_stats':True, 'tune-fc':False, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.2, 0.2, 0, 0], 'target_ch':4,'w_recon': 1, 'w_corr_sh': 12, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': True, 'freeze_stats':True, 'tune-fc':False, 'all-net':False},
        # {'batch_size':64, 'w_t':[0.1, 0.1, 0, 0], 'target_ch':4,'w_recon': 1, 'w_corr_sh': 12, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'lr': 0.01, 'decay': True, 'freeze_stats':True, 'tune-fc':False, 'all-net':False},
        
]       


# n_tasks=2
runs_metrics = []
runs_correlations = []

runs_losses = []
for run_idx, run in enumerate(runs):
    
    models=[]
    for task in range(n_tasks):
        # models.append(initilize_resNet(out_classes=5, model='resnet18', pretrained=None, load='models/bincifar_res18_5way_t'+str(task)+'.ckp', device=device))
        models.append(initilize_resNet(out_classes=5, model='resnet18', pretrained=None, load=None, device=device))

    mtl_ft_data = BinaryCIFAR(root='data/processed/cifar/train', tasks=chosen_tasks, transform=transform)
    dload_mtl_ft = DataLoader(mtl_ft_data, batch_size=run['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    if dload_mtl_ft.dataset.requires_soft==False:
        print('Producing soft-labels task')
        dload_mtl_ft.dataset.produce_soft_labels(models, 128, device)



    optims = []
    for task in range(n_tasks):
        optims.append(torch.optim.SGD(models[task].parameters(), lr=run['lr'], momentum=0.9))

    fc_optims = []
    for task in range(n_tasks):
        fc_optims.append(torch.optim.SGD(models[task].fc.parameters(), lr=0.001, momentum=0.9))
    
    hooks_manager=[]
    for task in range(n_tasks):
        hooks_manager.append(Intermediate())

    if run['decay']:
        scheds = []
        for task in range(n_tasks):
            scheds.append(torch.optim.lr_scheduler.StepLR(optims[task], step_size=2, gamma=0.98))
            # scheds.append(torch.optim.lr_scheduler.MultiStepLR(optims[task], [30], gamma=0.3))
            # scheds.append(torch.optim.lr_scheduler.MultiStepLR(optims[task],[30, 50], gamma=0.5))
    
    run_losses = []
    run_shared_corr = []
    run_recon_loss = []


    ft_metrics = []

    train_acc = []
    ft_acc = []
    val_acc = []
    test_acc = []

    for task in range(n_tasks):
        train_acc.append(evaluate(models[task], dload_train[task], device=device))
        ft_acc.append(evaluate(models[task], dload_ft[task], device=device))
        val_acc.append(evaluate(models[task], dload_val[task], device=device))
        test_acc.append(evaluate(models[task], dload_test[task], device=device))
    ft_metrics.append([train_acc, ft_acc, val_acc, test_acc])
    


    for task in range(n_tasks):
        for param in models[task].parameters():
            param.requires_grad = run['all-net']

        if run['tune-bn']:
            models[task].apply(bn_requires_grad)

        if run['tune-fc']:
            for param in models[task].fc.parameters():
                param.requires_grad = True

    correlations = []

    # get intermediate representations dims
    h_shapes = []
    hooks_manager[0].register_hooks(models[0], BasicBlock)
    # forward pass of a batch
    models[0](iter(dload_mtl_ft).next()[0][0].to(device))
    tensors = hooks_manager[0].tensors
    n_layers = len(tensors)
    for t in tensors:
        h_shapes.append(t.shape)
    hooks_manager[0].remove_handles()

    hist_layers =[]
    for task in range(n_tasks):
        task_histograms = []
        for h in h_shapes:
            task_histograms.append(SoftHist(h[1]//2, h[2], h[3], bins=8, min_value=0-0.2, max_value=5, sigma=100, momentum=1).to(device))

        hist_layers.append(task_histograms)


    
    # building encoders
    task1_enc = []
    task2_enc = []
    task1_dec = []
    task2_dec = []
    for h in h_shapes:

        in_channels = h[1]//2
        target_channels = in_channels//run['target_ch']

        kernel_size = 1
        if run['spatial_pooling']:
            kernel_size = int(math.ceil(h[2]/2))
    
        
        task1_enc.append(nn.Conv2d(in_channels, target_channels, (kernel_size,kernel_size)).to(device))
        task2_enc.append(nn.Conv2d(in_channels, target_channels, (kernel_size,kernel_size)).to(device))

        task1_dec.append(nn.ConvTranspose2d(target_channels, in_channels, (kernel_size,kernel_size)).to(device))
        task2_dec.append(nn.ConvTranspose2d(target_channels, in_channels, (kernel_size,kernel_size)).to(device))

    
    autoenc_params_t1 = []
    for layer, enc in enumerate(task1_enc):
        autoenc_params_t1.extend(list(enc.parameters()) + list(task1_dec[layer].parameters()))

    autoenc_params_t2 = []
    for layer, enc in enumerate(task2_enc):
        autoenc_params_t2.extend(list(enc.parameters()) + list(task2_dec[layer].parameters()))

    autoenc_optims = []
    autoenc_optims.append(torch.optim.Adam(autoenc_params_t1, lr=0.001))
    autoenc_optims.append(torch.optim.Adam(autoenc_params_t2, lr=0.001))



    for epoch in range (mtlcorr_epochs):

        train_acc = []
        ft_acc = []
        val_acc = []
        test_acc = []
        
        for i, batch in enumerate(dload_mtl_ft):


            for task in range(n_tasks):
                # hooks_manager[task].register_hooks(models[task], torch.nn.modules.batchnorm.BatchNorm2d)
                hooks_manager[task].register_hooks(models[task], BasicBlock)

                if run['freeze_stats']:
                    for m in models[task].parameters():
                        if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
                            m.eval()


            

            loss, losses, shared_corr, spec_corr, spar_loss, soft_losses, gather_correlations, reconstruction_loss = train_mtl_corr(batch, 
                                                                                                                models=models,
                                                                                                                criterions=criterions, 
                                                                                                                optims=optims, 
                                                                                                                weights=run,
                                                                                                                hooks_manager=hooks_manager)
            run_losses.append(loss.item())
            if run['w_corr_sh'] > 0:
                run_shared_corr.append(shared_corr.item())
                if run['w_recon'] > 0:
                    run_recon_loss.append(reconstruction_loss.item())

            if not gather_correlations:
                gather_correlations = [0]* n_layers

            correlations.append(gather_correlations)
                                                                                               
            for task in range(n_tasks):
                hooks_manager[task].remove_handles()

            print('[MTL-corr][Run %d/%d][Epoch %d/%d][Batch %d/%d][Loss %f][Corr %f/%f][Reconstr %f][Spar %f]' % 
            (run_idx, len(runs), epoch, mtlcorr_epochs, i, len(dload_mtl_ft), loss, shared_corr, spec_corr, reconstruction_loss, spar_loss))


        for task in range(n_tasks):
            train_acc.append(evaluate(models[task], dload_train[task], device=device))
            ft_acc.append(evaluate(models[task], dload_ft[task], device=device))
            val_acc.append(evaluate(models[task], dload_val[task], device=device))
            test_acc.append(evaluate(models[task], dload_test[task], device=device))

        ft_metrics.append([train_acc, ft_acc, val_acc, test_acc])
        if run['decay']:
            for task in range(n_tasks):
                scheds[task].step()
        
    runs_correlations.append(correlations)
    runs_metrics.append(ft_metrics)

    runs_losses.append([run_losses, run_shared_corr, run_recon_loss])


print('correlation shape: ', np.array(runs_correlations).shape)


data = {'train_metrics':train_metrics, 
        'run_metrics':runs_metrics,
        'correlations':runs_correlations,
        'losses': runs_losses,
        'runs':runs}


with open(out_dir/'bincifar_metrics.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


for task in range(n_tasks):
    gammas = []
    betas = []
    for m in models[task].modules():
        if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
            for name,param in m.named_parameters():
                if name == 'weight':
                    gammas.append(param.detach().cpu().numpy().flatten())
                else:
                    betas.append(param.detach().cpu().numpy().flatten())

    gammas = np.concatenate(gammas, axis=0)
    betas = np.concatenate(betas, axis = 0)

    plt.hist(gammas, bins=150, density=True)
    plt.savefig(out_dir/('bincifar_gammas_t'+str(task)+'.png'))
    plt.close()

    plt.hist(betas, bins=150, density=True, stacked=True)
    plt.savefig(out_dir/('bincifar_betas_t'+str(task)+'.png'))
    plt.close()


metrics = np.array(runs_metrics)
print(metrics.shape) #(n_runs, n_epochs, n_metrics, n_models)

correlations = np.array(correlations)
print(correlations.shape)
for net_cor in range(correlations.shape[1]):
    plt.plot(correlations[:, net_cor], alpha=0.6, label=str(net_cor))

plt.legend(loc='upper left')
plt.savefig(out_dir/'bincifar_correlations.png')
plt.close()



# for i, run in enumerate(runs):
#     plt.plot(metrics[i, :, -1, 0], label=str(i))
# plt.legend(loc='lower left')
# plt.savefig('bincif_t1.png')
# plt.close()

for task in range(n_tasks):
    for i, run in enumerate(runs):
        plt.plot(metrics[i, :, -2, task], label=str(i))

    plt.legend(loc='lower left')
    plt.savefig(out_dir/('bincifar_val_t'+str(task)+'.png'))
    plt.close()

for task in range(n_tasks):
    for i, run in enumerate(runs):
        plt.plot(metrics[i, :, -1, task], label=str(i))
    plt.legend(loc='lower left')
    plt.savefig(out_dir/('bincifar_test_t'+str(task)+'.png'))
    plt.close()


for task in range(n_tasks):
    # plotting learning curves
    plt.plot(metrics[0, :, -2, task], label='val')
    plt.plot(metrics[0, :, 0, task], label='train')
    plt.plot(metrics[0, :, 1, task], label='ft')
    plt.legend(loc='lower left')
    plt.savefig(out_dir/('bincifar_lc_t'+str(task)+'.png'))
    plt.close()

print('Finished')


