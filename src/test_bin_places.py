import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.models import resnet18, vgg
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import ToTensor, Compose, Normalize, RandomResizedCrop, CenterCrop
from utils.correlation import max_corr, soft_max_corr
from utils.model import initilize_resNet, Intermediate
from model.baseline import ShallowLinear, FilmLayer
import matplotlib.pyplot as plt
from data.dataset import CIFARDataset, MultitaskLabels, BinaryCIFAR, Places365, imshow
import pickle
import numpy as np
import random
import seaborn as sns

from PIL import Image

torch.manual_seed(5)
random.seed(6)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

pre_train_epochs = 0
mtlcorr_epochs = 70
n_tasks = 2

# def get_intermediate(self, task):
#     def hook(model, input, output):
#         intermediate[task].append(output)
#     return hook

# hooks =[]
# def register_hooks(m):
#     if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
#         hooks.append({'handles':[], 'intermediate':[]})

#         handles.append(m.register_forward_hook(get_intermediate))
#         m.eval()



# intermediate_tensors = []
# def save_tensor(self, input, output):
#     intermediate_tensors.append(output[:, int(output.shape[1]//2), :, :])

# handles = []
# def save_intermediate(m):
#     if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
#         handles.append(m.register_forward_hook(save_tensor))

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


def train_mtl_corr(batches, models, criterions, optims, weights, hooks_manager):

    loss_weights = weights['w_t']
    w_corr_sh = weights['w_corr_sh']
    w_corr_spec = weights['w_corr_spec']
    w_spar = weights['w_spar']
    soft_loss_weights = weights['w_soft']

    T = 20

    losses = []
    soft_losses = []
    for task, model in enumerate(models):
        
        optims[task].zero_grad()
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
    if w_corr_sh != 0:
        n_tasks = len(models)
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
                    shape = hooks_manager[s].tensors[layer].shape
                    shared_s = source[layer][:, :int(shape[1]//2), :, :]
                    shared_t = target[layer][:, :int(shape[1]//2), :, :]
                    print('min %f, max %f, %f, %f' % (shared_s.min().item(), shared_s.max().item(), shared_t.min().item(), shared_t.max().item()))
                    shared_corr = shared_corr + max_corr(shared_s, shared_t, device)

                    # print('soft_corr', soft_max_corr(shared_s.detach(), shared_t.detach(), device))
                    # print('max_corr', max_corr(shared_s.detach(), shared_t.detach(), device))

                    # max_corr_1 = max_corr_1 + max_corr(shared_s, shared_t.detach(), device)
                    # max_corr_2 = max_corr_2 + max_corr(shared_s.detach(), shared_t, device)
                    corr = max_corr(shared_s, shared_t, device).item()
                    # shared_corr = shared_corr + corr
                    gather_correlations.append(corr)
     

        shared_corr = shared_corr/n_layers
        max_corr_1 = max_corr_1 / n_layers
        max_corr_2 = max_corr_2 / n_layers




    # for each task at each layer compute the correlation between specidic subspaces
    spec_corr = 0
    if w_corr_spec != 0:            
        for task in range(n_tasks):
            for layer in range(n_layers):
                shape =  hooks_manager[task].tensors[layer].shape
                sh = hooks_manager[task].tensors[layer][:, :int(shape[1]//2), :, :]
                spec = hooks_manager[task].tensors[layer][:, int(shape[1]//2):, :, :]
                spec_corr = spec_corr + max_corr(sh, spec, device)
        spec_corr = spec_corr / (n_tasks)



    spar_loss = 0
    if w_spar != 0:
        spar_loss = spar(models)

    corr = spec_corr - shared_corr

    w_tot_loss =0
    w_tot_soft_loss = 0
    for task in range(len(losses)):
        w_tot_loss = w_tot_loss +  loss_weights[task]*losses[task]
        w_tot_soft_loss = w_tot_soft_loss + soft_loss_weights[task]*soft_losses[task]
    
 
    loss = w_tot_loss + w_corr_sh*corr + w_spar*spar_loss + w_tot_soft_loss

    loss.backward()
    for opt in optims:
        opt.step()

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

    
    return loss, losses, shared_corr, spec_corr, spar_loss, soft_losses, gather_correlations

orig_models=[]
for task in range(n_tasks):
    orig_model = initilize_resNet(out_classes=2, pretrained=None, load=None, device=device)
    orig_models.append(dict(orig_model.named_parameters()))

def spar(models):

    tot_spar = 0
    for task, model in enumerate(models):
        for name, param in model.named_parameters():
            tot_spar = tot_spar + torch.norm((orig_models[task][name]- param), 2)
    return tot_spar/len(models)

    # tot_spar = 0
    # for task, model in enumerate(models):
    #     for name, param in model.fc.named_parameters():
    #         tot_spar = tot_spar + torch.norm((orig_models[task][name]- param), 2)
    # return tot_spar/len(models)
    

    # spar_t2 = 0
    # for name, param in model_t2.named_parameters():
    #     spar_t2 = spar_t2 + torch.norm((orig_model_t2[name]- param), 2)

    # return spar_t1 + spar_t2


# transform=ToTensor()
transform = Compose([RandomResizedCrop(64), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform = Compose([CenterCrop(64), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


dload_train = []
dload_ft = []
dload_val = []
dload_test = []

chosen_tasks = []
for task in range(n_tasks):
    
    task = random.choices(range(365), k=2)
    chosen_tasks.append(task)

    train_data = Places365('data/processed/mini-places365', image_set='train', tasks=[task], transform=transform)
    dload_train.append(DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4))

    ft_data = Places365('data/processed/mini-places365', image_set='ft', tasks=[task], transform=transform)
    dload_ft.append(DataLoader(ft_data, batch_size=64, shuffle=True, num_workers=4, drop_last=False))
    
    val_data = Places365('data/processed/mini-places365', image_set='val', tasks=[task], transform=test_transform)
    dload_val.append(DataLoader(val_data, batch_size=128, shuffle=False, num_workers=4))

    test_data = Places365('data/processed/mini-places365', image_set='test', tasks=[task], transform=test_transform)
    dload_test.append(DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4))

print(chosen_tasks)
mtl_ft_data = Places365('data/processed/mini-places365', image_set='ft', tasks=chosen_tasks, transform=transform)
dload_mtl_ft = DataLoader(mtl_ft_data, batch_size=64, shuffle=True, num_workers=4, drop_last=False)


models=[]
for task in range(n_tasks):
    models.append(initilize_resNet(out_classes=2, model='resnet18', pretrained=False, load=None, device=device))

criterions = []
for task in range(n_tasks):
    criterions.append(torch.nn.CrossEntropyLoss().to(device))


optims = []
for task in range(n_tasks):
    optims.append(torch.optim.SGD(models[task].parameters(),lr=0.1, momentum=0.9, weight_decay=5e-4))

scheds = []
for task in range(n_tasks):
    scheds.append(torch.optim.lr_scheduler.MultiStepLR(optims[task], [60, 100], gamma=0.1))


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

            pred = models[task](sample)
            loss = criterions[task](pred, labels)

            loss.backward()
            optims[task].step()

            print('[PRE-TRAIN][TASK %d/%d][Epoch %d/%d][Batch %d/%d][Loss %f]' % (task, n_tasks, epoch, pre_train_epochs, i, len(dload_train[task]), loss))


        train_acc.append(evaluate(models[task], dload_train[task], device=device))
        ft_acc.append(evaluate(models[task], dload_ft[task], device=device))
        val_acc.append(evaluate(models[task], dload_val[task], device=device))
        test_acc.append(evaluate(models[task], dload_test[task], device=device))
        
        
        
        scheds[task].step()

    train_metrics.append([train_acc, ft_acc, val_acc, test_acc])

train_metrics = np.array(train_metrics)
print(train_metrics.shape) # (n_tasks, n_metrics, n_epochs)

# for task in range(n_tasks):
#     plt.plot(metrics[task, -1, :], label='test')
#     plt.plot(metrics[task, 0, :], label='train')
#     plt.legend(loc='lower left')
#     plt.savefig('binplaces_training_t'+str(task)+'.png')
#     plt.close()

# for task in range(n_tasks):
#     print(metrics[task, -1, :])

data = {'train_metrics':train_metrics}

# with open('binplaces_train_metrics.pkl', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print('Saving models')
# for task, model in enumerate(models):
#     torch.save(model.state_dict(), 'binplaces_res18_t'+str(task)+'.ckp')


# raise SystemError(0)

# runs = []
# corr = [0, 0.1, 0.2, 0.5, 1, 2, 4]
# soft = [0, 0.1, 0.2, 0.4]
# for wc in corr:
#     for ws in soft:
#         runs.append({'w_t':[0.25, 0.25], 'w_corr': wc, 'w_spar': 0, 'w_soft': ws, 'bn_ft': True, 'lr': 0.001, 'decay': False})
runs = [
        # {'w_t':[0.5, 0.5, 0, 0], 'w_corr_sh': 0.4, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0.1, 0.1, 0, 0], 'bn_ft': True, 'lr': 0.001, 'decay': False, 'freeze_stats':False, 'tune-fc':True},
        {'w_t':[0, 0, 0, 0], 'w_corr_sh': 0.5, 'w_corr_spec': 0, 'w_spar': 0, 'w_soft': [0, 0, 0, 0], 'bn_ft': True, 'lr': 0.01, 'decay': False, 'freeze_stats':False, 'tune-fc':True},
]       


# n_tasks=2
runs_metrics = []
for run_idx, run in enumerate(runs):
    
    models=[]
    for task in range(n_tasks):
        models.append(initilize_resNet(out_classes=2, model='resnet18', pretrained=None, load='models/binplaces_res18_t'+str(task)+'.ckp', device=device))
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
            # scheds.append(torch.optim.lr_scheduler.MultiStepLR(optims[task], [60, 100], gamma=0.1))
            scheds.append(torch.optim.lr_scheduler.MultiStepLR(optims[task],[30, 50], gamma=0.5))

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
    

    if run['bn_ft']:
        for task in range(n_tasks):
            for param in models[task].parameters():
                param.requires_grad = False

            models[task].apply(bn_requires_grad)
            if run['tune-fc']:
                for param in models[task].fc.parameters():
                    param.requires_grad = True

    correlations = []
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

            

            loss, losses, shared_corr, spec_corr, spar_loss, soft_losses, gather_correlations = train_mtl_corr(batch, 
                                                                                                                models=models,
                                                                                                                criterions=criterions, 
                                                                                                                optims=optims, 
                                                                                                                weights=run,
                                                                                                                hooks_manager=hooks_manager)
            correlations.append(gather_correlations)                                                                                            

            for task in range(n_tasks):
                hooks_manager[task].remove_handles()

            print('[MTL-corr][Run %d/%d][Epoch %d/%d][Batch %d/%d][Loss %f][Corr %f/%f][Spar %f]' % 
            (run_idx, len(runs), epoch, mtlcorr_epochs, i, len(dload_mtl_ft), loss, shared_corr, spec_corr, spar_loss))


        for task in range(n_tasks):
            train_acc.append(evaluate(models[task], dload_train[task], device=device))
            ft_acc.append(evaluate(models[task], dload_ft[task], device=device))
            val_acc.append(evaluate(models[task], dload_val[task], device=device))
            test_acc.append(evaluate(models[task], dload_test[task], device=device))

        ft_metrics.append([train_acc, ft_acc, val_acc, test_acc])
        if run['decay']:
            for task in range(n_tasks):
                scheds[task].step()
        

    runs_metrics.append(ft_metrics)



data = {'train_metrics':train_metrics, 
        'run_metrics':runs_metrics,
        'runs':runs}


with open('binplaces_metrics.pkl', 'wb') as handle:
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
    plt.savefig('binplaces_gammas_t'+str(task)+'.png')
    plt.close()

    plt.hist(betas, bins=150, density=True, stacked=True)
    plt.savefig('binplaces_betas_t'+str(task)+'.png')
    plt.close()


metrics = np.array(runs_metrics)
print(metrics.shape) #(n_runs, n_epochs, n_metrics, n_models)

correlations = np.array(correlations)
print(correlations.shape)
for net_cor in range(correlations.shape[1]):
    plt.plot(correlations[:, net_cor], alpha=0.6, label=str(net_cor))

plt.savefig('binplaces_correlations.png')
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
    plt.savefig('binplaces_t'+str(task)+'.png')
    plt.close()

for task in range(n_tasks):
    for i, run in enumerate(runs):
        plt.plot(metrics[i, :, -1, task], label=str(i))
    plt.legend(loc='lower left')
    plt.savefig('binplaces_test_t'+str(task)+'.png')
    plt.close()


for task in range(n_tasks):
    # plotting learning curves
    plt.plot(metrics[0, :, -2, task], label='val')
    plt.plot(metrics[0, :, 0, task], label='train')
    plt.plot(metrics[0, :, 1, task], label='ft')
    plt.legend(loc='lower left')
    plt.savefig('binplaces_lc_t'+str(task)+'.png')
    plt.close()

print('Finished')


