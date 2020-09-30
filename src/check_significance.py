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
from model.baseline import ShallowLinear, FilmLayer,soft_hist, GaussianHistogram
import matplotlib.pyplot as plt
from data.dataset import CIFARDataset, MultitaskLabels, BinaryCIFAR, Places365, imshow
import pickle
import numpy as np
import random
import seaborn as sns
import math

from PIL import Image

torch.manual_seed(5)
random.seed(5)
np.random.seed(5)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

pre_train_epochs = 0
mtlcorr_epochs = 70
n_tasks = 2

def bn_requires_grad(m):

    if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
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


def channels_divergence(batches, models, hooks_manager):

    # load data
    samples_s = batches[0][0].to(device)
    pred = models[0](samples_s)

    samples_t = batches[1][0].to(device)
    pred = models[1](samples_t)
    
    n_layers = len(hooks_manager[0].tensors)
    source = hooks_manager[0].tensors
    target = hooks_manager[1].tensors

    # for each layer compute the correlation between shared subspaces
    net_divs = []
    for layer in range(n_layers):
        print('Computing layer %d/%d' % (layer, n_layers))
        shape = hooks_manager[0].tensors[layer].shape
        shared_s = source[layer][:, :int(shape[1]//2), :, :]
        shared_t = target[layer][:, :int(shape[1]//2), :, :]

        # let's compute how different are feature maps
        n_channels = shape[1]//2
        l_divs = torch.empty((n_channels, n_channels))
        bins=8
        v_min, v_max = shared_s[:, :].min().item(), shared_s[:, :].max().item()
        for ch_idx in range(n_channels):
            for ch_idy in range(n_channels):
                n_elements = np.prod(shared_s[:, ch_idx].shape)

                feat_map_x = shared_s[:, ch_idx]
                feat_map_y = shared_s[:, ch_idy]  

                x_hist = torch.add(torch.histc(feat_map_x.reshape(-1), bins=bins, min=v_min, max=v_max), 1.)
                x_hist = torch.true_divide(x_hist, n_elements + bins).unsqueeze(dim=0)

                y_hist = torch.add(torch.histc(feat_map_y.reshape(-1), bins=bins, min=v_min, max=v_max), 1.)
                y_hist = torch.true_divide(y_hist, n_elements + bins).unsqueeze(dim=0)

                kl_div = F.kl_div(y_hist.log(), x_hist)
        
                l_divs[ch_idx, ch_idy] = kl_div.item()


        print(torch.max(torch.sum(l_divs, dim=1), dim=0)[1])
        # print(torch.max(torch.sum(l_divs, dim=0))[1])
        net_divs.append(l_divs)
    
    
    return net_divs

def get_statistics_noiseness(batches, models, hooks_manager):

    # load data
    samples = batches[0][0].to(device)

    pred = models[0](samples)
    
    n_layers = len(hooks_manager[0].tensors)
    source = hooks_manager[0].tensors

    # for each layer compute the correlation between shared subspaces
    net_divs = []
    
    channel = 0
    bins = 8
    model_var = []
    for layer in range(n_layers):
        v_min, v_max = source[layer][:, channel, :, :].min().item(), source[layer][:, channel, :, :].max().item()
        
        shape = hooks_manager[0].tensors[layer].shape
        # n_elements = np.prod(source[layer][0, channel, :, :].shape)

        l_var = []
        for sample in range(1, samples.shape[0]-1):

            running_feat = source[layer][:sample, channel, :, :]
            running_feat_n_elem = np.prod(running_feat.shape)

            feat_next = source[layer][:(sample + 1), channel, :, :]
            feat_next_n_elem = np.prod(feat_next.shape)

            running_feat_hist = torch.add(torch.histc(running_feat.reshape(-1), bins=bins, min=v_min, max=v_max), 1.)
            running_feat_hist = torch.true_divide(running_feat_hist, running_feat_n_elem + bins).unsqueeze(dim=0)

            feat_next_hist = torch.add(torch.histc(feat_next.reshape(-1), bins=bins, min=v_min, max=v_max), 1.)
            feat_next_hist = torch.true_divide(feat_next_hist, feat_next_n_elem + bins).unsqueeze(dim=0)

            kl_div = F.kl_div(feat_next_hist.log(), running_feat_hist)

            l_var.append(kl_div.item())

        model_var.append(l_var)

    return np.array(model_var)


orig_models=[]
for task in range(n_tasks):
    orig_model = initilize_resNet(out_classes=2, model='resnet18', pretrained=None, load=None, device=device)
    orig_models.append(dict(orig_model.named_parameters()))

def spar(models):

    tot_spar = 0
    for task, model in enumerate(models):
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


chosen_tasks = [[62, 74, 79, 94, 73], [99, 79, 46, 49, 93]]

mtl_ft_data = BinaryCIFAR(root='data/processed/cifar/train', tasks=chosen_tasks, transform=transform)
dload_mtl_ft = DataLoader(mtl_ft_data, batch_size=1900, shuffle=True, num_workers=4, drop_last=True)


runs = [
        {'w_t':[0.5, 0.5], 'w_corr_sh': 25, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0], 'bn_ft': True, 'lr': 0.01, 'decay': True, 'freeze_stats':False, 'tune-fc':False},
]       


# n_tasks=2
runs_metrics = []
for run_idx, run in enumerate(runs):
    
    models=[]
    for task in range(n_tasks):
        models.append(initilize_resNet(out_classes=5, model='resnet18', pretrained=None, load='models/bincifar_res18_5way_t'+str(task)+'.ckp', device=device))
        # models.append(initilize_resNet(out_classes=2, pretrained=None, load=None, device=device))

 
    if dload_mtl_ft.dataset.requires_soft==False:
        print('Producing soft-labels task')
        dload_mtl_ft.dataset.produce_soft_labels(models, 128, device)

    optims = []

    for task in range(n_tasks):
        optims.append(torch.optim.SGD(models[task].parameters(), lr=run['lr'], momentum=0.9))
        # {'params':models[task].fc.parameters(), 'lr':0.001}
    
    hooks_manager=[]
    for task in range(n_tasks):
        hooks_manager.append(Intermediate())

    if run['decay']:
        scheds = []
        for task in range(n_tasks):
            # scheds.append(torch.optim.lr_scheduler.StepLR(optims[task], step_size=2, gamma=0.98))
            scheds.append(torch.optim.lr_scheduler.MultiStepLR(optims[task], [30], gamma=0.3))
            # scheds.append(torch.optim.lr_scheduler.MultiStepLR(optims[task],[30, 50], gamma=0.5))

    
    if run['bn_ft']:
        for task in range(n_tasks):
            for param in models[task].parameters():
                param.requires_grad = False

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
    for t in tensors:
        h_shapes.append(t.shape)
    hooks_manager[0].remove_handles()

    
    # building encoders
    task1_enc = []
    task2_enc = []
    task1_dec = []
    task2_dec = []
    for h in h_shapes:

        in_channels = h[1]//2
        target_channels = in_channels//8
        kernel_size = int(math.ceil(h[2]/2))
        task1_enc.append(nn.Sequential(nn.Conv2d(in_channels, target_channels, (kernel_size,kernel_size)).to(device),
                        ))
        task2_enc.append(nn.Sequential(nn.Conv2d(in_channels, target_channels, (kernel_size,kernel_size)).to(device),
                        ))

        task1_dec.append(nn.ConvTranspose2d(target_channels, in_channels, (kernel_size,kernel_size)).to(device))
        task2_dec.append(nn.ConvTranspose2d(target_channels, in_channels, (kernel_size,kernel_size)).to(device))


    # histograms
    hists_t1 = []
    hists_t2 = []

    for h in range(len(h_shapes)):
        hists_t1.append(GaussianHistogram(16, min=0, max=5, sigma=100).to(device))
        hists_t2.append(GaussianHistogram(16, min=0, max=5, sigma=100).to(device))

    
    for i, batch in enumerate(dload_mtl_ft):
        print(i)
        for task in range(n_tasks):
            hooks_manager[task].register_hooks(models[task], BasicBlock)

            if run['freeze_stats']:
                for m in models[task].parameters():
                    if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
                        m.eval()

        net_divs = channels_divergence(batch, models, hooks_manager)     
        model_var = get_statistics_noiseness(batch, models, hooks_manager)


        fig, ax = plt.subplots()
        for layer in range(model_var.shape[0]):
            ax.plot(model_var[layer, :], label = str(layer))
        
        axins = ax.inset_axes([0.2, 0.5, 0.65, 0.47])
        for layer in range(model_var.shape[0]):
            axins.plot(model_var[layer, 20:], label = str(layer))
        
        # sub region of the original image
        x1, x2, y1, y2 = 20, 70, 0, 0.00005
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        plt.legend()
        plt.savefig('convergence of statistics.png')
        plt.close()


        for task in range(n_tasks):
            hooks_manager[task].remove_handles()


        ax = sns.heatmap(net_divs[1].cpu().numpy(), cmap='Reds')
        plt.title('T1-Correlating layer 1: feature divergences')
        plt.savefig('same_task_KL_div_layer1_t0.png')
        plt.close()
        
        ax = sns.heatmap(net_divs[-1].cpu().numpy(), cmap='Reds')
        plt.title('T1-Correlating layer 7: feature divergences')
        plt.savefig('same_task_KL_div_layer7_t0.png')
        plt.close()


    raise SystemError(0)

data = {'train_metrics':train_metrics, 
        'run_metrics':runs_metrics,
        'correlations':correlations,
        'runs':runs}


with open('bincifar_metrics.pkl', 'wb') as handle:
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
    plt.savefig('bincifar_gammas_t'+str(task)+'.png')
    plt.close()

    plt.hist(betas, bins=150, density=True, stacked=True)
    plt.savefig('bincifar_betas_t'+str(task)+'.png')
    plt.close()


metrics = np.array(runs_metrics)
print(metrics.shape) #(n_runs, n_epochs, n_metrics, n_models)

correlations = np.array(correlations)
print(correlations.shape)
for net_cor in range(correlations.shape[1]):
    plt.plot(correlations[:, net_cor], alpha=0.6, label=str(net_cor))

plt.legend(loc='upper left')
plt.savefig('bincifar_correlations.png')
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
    plt.savefig('bincifar_val_t'+str(task)+'.png')
    plt.close()

for task in range(n_tasks):
    for i, run in enumerate(runs):
        plt.plot(metrics[i, :, -1, task], label=str(i))
    plt.legend(loc='lower left')
    plt.savefig('bincifar_test_t'+str(task)+'.png')
    plt.close()


for task in range(n_tasks):
    # plotting learning curves
    plt.plot(metrics[0, :, -2, task], label='val')
    plt.plot(metrics[0, :, 0, task], label='train')
    plt.plot(metrics[0, :, 1, task], label='ft')
    plt.legend(loc='lower left')
    plt.savefig('bincifar_lc_t'+str(task)+'.png')
    plt.close()

print('Finished')


