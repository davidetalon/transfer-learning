import torch
import torch.nn as nn

from torchvision.transforms import Compose, ToTensor, Resize, Normalize,RandomHorizontalFlip, RandomCrop, Pad
from torchvision.models import resnet34, resnet18, alexnet, mobilenet_v2
from torchvision.datasets import ImageNet, CIFAR100
from torch.utils.data import DataLoader
from data.dataset import ImageDataset, CIFARDataset, MultitaskLabels
from model.baseline import ShallowLinear
import numpy as np

from utils.correlation import max_corr, soft_max_corr
from utils.model import evaluate, bn_requires_grad, initilize_resNet
import matplotlib.pyplot as plt


pre_training_epochs= 0
fine_tuning_epochs = 0
mtlcorr_epochs = 50
mtl_epochs = 0
torch.manual_seed(4)
np.random.seed(5)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
runs=1

tensors=[]
def get_intermediate(self, input, output):
    tensors.append(output[:, :int(output.shape[1]//2), :, :])

def train_mtl_corr(batch, models, losses, optimizers):

    optimizers[0].zero_grad()
    optimizers[1].zero_grad()

    samples = batch[0].to(device)
    labels = batch[1].to(device)

    # predict
    if torch.isnan(samples).sum():
        raise SystemError(0)

    pred = models[0](samples)
    t1_loss = losses[0](pred, labels[:, 0])


    pred = models[1](samples)
    t2_loss = losses[1](pred, labels[:, 1])

    # corr = torch.zeros(36).to(device)

    corr = 0
    for param in models[1].parameters():
        if torch.isnan(param).sum():
            raise SystemError(0)

    for layer in range(20):
        corr= corr + max_corr(tensors[layer], tensors[20 + layer], device)


    # corr = weightning(corr)

    # dist = 0
    # for layer in range(20):
    #     dist = dist + torch.pow(torch.norm(tensors[layer]-tensors[20+layer]), 2)

    loss= t1_loss + t2_loss - corr
    loss.backward()

    

    optimizers[0].step()
    optimizers[1].step()

    return t1_loss, t2_loss, corr

class WeightsModel(nn.Module):

    def __init__(self, dim):

        super(WeightsModel, self).__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = torch.dot(self.alpha, x)
        return x
        




        
# def bn_requires_grad(m):
#     if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
#         # m.reset_running_stats()
#         for param in m.parameters():
#             param.requires_grad = True



handles = []
def register_hooks(m):
    if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
        handles.append(m.register_forward_hook(get_intermediate))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# transform = Compose([Pad(4, fill=0, padding_mode='constant'), 
#                     RandomCrop(32, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
#                     RandomHorizontalFlip(p=0.5),
#                     ToTensor(),  
#                     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform=ToTensor()

# task1
task1 = MultitaskLabels([range(0,50,1)])
train_t1 = CIFARDataset(root='data/processed/cifar/train', transform=transform, target_transform=task1)
dloader_train_t1 = torch.utils.data.DataLoader(train_t1, batch_size=128, shuffle=True, num_workers=4)

ft_t1 = CIFARDataset(root='data/processed/cifar/fine-tune', transform=transform, target_transform=task1)
dloader_ft_t1 = torch.utils.data.DataLoader(ft_t1, batch_size=128, shuffle=True, num_workers=4)

test_t1 = CIFARDataset(root='data/processed/cifar/test', transform=transform, target_transform=task1)
dloader_test_t1 = torch.utils.data.DataLoader(test_t1, batch_size=128, shuffle=False, num_workers=4)


# task2
task2 = MultitaskLabels([list(range(30, 50, 1)) + list(range(60,90, 1))])
train_t2 = CIFARDataset(root='data/processed/cifar/train', transform=transform, target_transform=task2)
dloader_train_t2 = torch.utils.data.DataLoader(train_t2, batch_size=128, shuffle=True, num_workers=4)

ft_t2 = CIFARDataset(root='data/processed/cifar/fine-tune', transform=transform, target_transform=task2)
dloader_ft_t2 = torch.utils.data.DataLoader(ft_t2, batch_size=128, shuffle=True, num_workers=4)

test_t2 = CIFARDataset(root='data/processed/cifar/test', transform=transform, target_transform=task2)
dloader_test_t2 = torch.utils.data.DataLoader(test_t2, batch_size=128, shuffle=False, num_workers=4)

# mtask
mtl = MultitaskLabels([range(0,50,1), list(range(30, 50, 1)) + list(range(60,90,1))])
train_mtask = CIFARDataset(root='data/processed/cifar/train', transform=transform, target_transform=mtl)
dloader_train_mtask = torch.utils.data.DataLoader(train_mtask, batch_size=128, shuffle=True, num_workers=4)

ft_mtask = CIFARDataset(root='data/processed/cifar/fine-tune', transform=transform, target_transform=mtl)
dloader_ft_mtask = torch.utils.data.DataLoader(ft_mtask, batch_size=16, shuffle=True, num_workers=4)

test_mtask = CIFARDataset(root='data/processed/cifar/test', transform=transform, target_transform=mtl)
dloader_test_mtask = torch.utils.data.DataLoader(test_mtask, batch_size=128, shuffle=False, num_workers=4)


snet = initilize_resNet(out_classes=2, pretrained=False, load=None, device=device)
tnet = initilize_resNet(out_classes=2, pretrained=False, load=None, device=device)

# optimizer
# opt_s = torch.optim.Adam(snet.parameters(), lr=0.01, betas=(0.9, 0.999))
opt_s = torch.optim.SGD(snet.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
opt_t = torch.optim.SGD(tnet.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
# scheduler_t = torch.optim.lr_scheduler.MultiStepLR(opt_t, milestones=[60,120,160], gamma=0.2)
# scheduler_t = torch.optim.lr_scheduler.MultiStepLR(opt_t, milestones=[150,225], gamma=0.1) #learning rate decay


# loss
t1_criterion = torch.nn.CrossEntropyLoss().to(device)
t2_criterion = torch.nn.CrossEntropyLoss().to(device)




test_acc_t1 = []
training_acc_t1 = []
for epoch in range (pre_training_epochs):
    for i, batch in enumerate(dloader_train_t1):

        opt_s.zero_grad()

        samples = batch[0].to(device)
        labels = batch[1].to(device)

        pred = snet(samples)

        loss = t1_criterion(pred, labels)
        loss.backward()

        # update parameters
        opt_s.step()

        print('[S][Epoch %d/%d][Batch %d/%d][Loss %f]' % (epoch, pre_training_epochs, i, len(dloader_train_t1), loss))
    
    # scheduler_s.step()

    # check for overfit/underfit
    training_acc_t1.append(evaluate(snet, dloader_train_t1, device=device))
    test_acc_t1.append(evaluate(snet, dloader_test_t1, device=device))

    # scheduler_s.step()

# print(training_acc)
# print(test_acc)

# plt.plot(training_acc_t1, 'k', label='training acc')
# plt.plot(test_acc_t1, 'g', label='test acc')
# plt.legend()
# plt.savefig('t1_training.png')

# raise SystemError(0)

# print("Saving source")
# torch.save(snet.state_dict(), "STL_t1_150epochs.ckp")

# evaluate task t1
# print('Evaluating task t1')
# stl_acc_train_t1 = evaluate(snet, dloader_test_t1)

for epoch in range (fine_tuning_epochs):
    for i, batch in enumerate(dloader_ft_t1):
        
        opt_s.zero_grad()

        samples = batch[0].to(device)
        labels = batch[1].to(device)


        pred = snet(samples)


        loss = t1_criterion(pred, labels.unsqueeze(-1).float())
        loss.backward()

        # update parameters
        opt_s.step()

        print('[S - FT][Epoch %d/%d][Batch %d/%d][Loss %f]' % (epoch, fine_tuning_epochs, i, len(dloader_ft_t1), loss))

training_acc_t2 = []
test_acc_t2 = []
for epoch in range (pre_training_epochs):
    for i, batch in enumerate(dloader_train_t2):

        opt_t.zero_grad()

        samples = batch[0].to(device)
        labels = batch[1].to(device)

        pred = tnet(samples)
        loss = t2_criterion(pred, labels)

        loss.backward()

        # update parameters
        opt_t.step()

        print('[T][Epoch %d/%d][Batch %d/%d][Loss %f]' % (epoch, pre_training_epochs, i, len(dloader_train_t2), loss))

    # scheduler_t.step()
    # check for overfit/underfit
    training_acc_t2.append(evaluate(tnet, dloader_train_t2, device=device))
    test_acc_t2.append(evaluate(tnet, dloader_test_t2, device=device))

# print(training_acc)
# print(test_acc)

# plt.plot(training_acc_t2, 'k', label='training acc')
# plt.plot(test_acc_t2, 'g', label='test acc')
# plt.legend()
# plt.savefig('t2_training.png')


# print("Saving target")
# torch.save(tnet.state_dict(), "STL_t2_150epochs.ckp")


# print('completed')
# raise SystemError(0)

# print('Evaluating task t1')
# stl_acc_train_t1 = evaluate(snet, dloader_test_t1)
# print('Evaluating task t2')
# stl_acc_train_t2 = evaluate(tnet, dloader_test_t2)

# print("Accuracy with trained models: ", stl_acc_train_t1, stl_acc_train_t2)


for epoch in range (fine_tuning_epochs):
    for i, batch in enumerate(dloader_ft_t2):

        opt_t.zero_grad()

        samples = batch[0].to(device)
        labels = batch[1].to(device)

        pred = tnet(samples)
        loss = t2_criterion(pred, labels.unsqueeze(-1).float())

        loss.backward()

        # update parameters
        opt_t.step()

        print('[T-FT][Epoch %d/%d][Batch %d/%d][Loss %f]' % (epoch, fine_tuning_epochs, i, len(dloader_ft_t2), loss))


# evaluate single-tasks models after fine-tuning
# print('Evaluating task t1 after fine-tuning')
# stl_acc_ft_t1 = evaluate(snet, dloader_test_t1)

# print('Evaluating task t2 after fine-tuning')
# stl_acc_ft_t2 = evaluate(tnet, dloader_test_t2)



##################################################################
# load saved models
##################################################################
# source model
snet = initilize_resNet(out_classes=2, pretrained=False, load="models/STL_t1_150epochs.ckp", device=device)
tnet = initilize_resNet(out_classes=2, pretrained=False, load="models/STL_t2_150epochs.ckp", device=device)
snet.apply(bn_requires_grad)
tnet.apply(bn_requires_grad)




opt_s = torch.optim.Adam(snet.parameters(), lr=0.001, betas=(0.9, 0.999))
opt_t = torch.optim.Adam(tnet.parameters(), lr=0.001, betas=(0.9, 0.999))

# scheduler_s = torch.optim.lr_scheduler.StepLR(opt_s, step_size=3, gamma=0.98)
# scheduler_t = torch.optim.lr_scheduler.StepLR(opt_t, step_size=3, gamma=0.98)
running_ft_acc_t1 = []
running_ft_acc_t1.append(evaluate(snet, dloader_test_t1, device=device))
for epoch in range (0):
    for i, batch in enumerate(dloader_ft_t1):
        
        opt_s.zero_grad()

        samples = batch[0].to(device)
        labels = batch[1].to(device)

        pred = snet(samples)

        loss = t1_criterion(pred, labels)
        loss.backward()

        # update parameters
        opt_s.step()

        print('[S - FT][Epoch %d/%d][Batch %d/%d][Loss %f]' % (epoch, mtlcorr_epochs, i, len(dloader_ft_t1), loss))
    running_ft_acc_t1.append(evaluate(snet, dloader_test_t1, device=device))

running_ft_acc_t2 = []
running_ft_acc_t2.append(evaluate(tnet, dloader_test_t2, device=device))
for epoch in range (0):
    for i, batch in enumerate(dloader_ft_t2):

        opt_t.zero_grad()

        samples = batch[0].to(device)
        labels = batch[1].to(device)

        pred = tnet(samples)
        loss = t2_criterion(pred, labels)

        loss.backward()

        # update parameters
        opt_t.step()

        print('[T-FT][Epoch %d/%d][Batch %d/%d][Loss %f]' % (epoch, mtlcorr_epochs, i, len(dloader_ft_t2), loss))
    running_ft_acc_t2.append(evaluate(tnet, dloader_test_t2, device=device))



##################################################################
# correlation mtl
##################################################################
mtl_corr_run_acc_t1 = []
mtl_corr_run_acc_t2 = []
for run in range(runs):

    snet = initilize_resNet(out_classes=2, pretrained=False, load="models/STL_t1_150epochs.ckp", device=device)
    tnet = initilize_resNet(out_classes=2, pretrained=False, load="models/STL_t2_150epochs.ckp", device=device)
    opt_s = torch.optim.Adam(snet.parameters(), lr=0.001, betas=(0.9, 0.999))
    opt_t = torch.optim.Adam(tnet.parameters(), lr=0.001, betas=(0.9, 0.999))


    for handle in handles:
        handle.remove()

    ft_acc_t1 = []
    ft_acc_t1.append(evaluate(snet, dloader_ft_t1, device=device))
    ft_acc_t2 = []
    ft_acc_t2.append(evaluate(tnet, dloader_ft_t2, device=device))

    running_test_acc_t1 = []
    running_test_acc_t1.append(evaluate(snet, dloader_test_t1, device=device))
    running_test_acc_t2 = []
    running_test_acc_t2.append(evaluate(tnet, dloader_test_t2, device=device))
    snet.apply(register_hooks)
    tnet.apply(register_hooks)


    # freeze all weights but BN and final classifier
    for param in snet.parameters():
        param.requires_grad = False
    snet.apply(bn_requires_grad)
    for param in snet.fc.parameters():
        param.requires_grad = True

    for param in tnet.parameters():
        param.requires_grad = False
    tnet.apply(bn_requires_grad)
    for param in tnet.fc.parameters():
        param.requires_grad = True

 

    for epoch in range(mtlcorr_epochs):
        for i, batch in enumerate(dloader_ft_mtask):

            t1_loss, t2_loss, corr = train_mtl_corr(batch, models=[snet, tnet], losses=[t1_criterion, t2_criterion], optimizers=[opt_s, opt_t])
            loss = t1_loss + t2_loss - corr
            tensors = []

            print('[MTL-corr][Run %d/%d][Epoch %d/%d][Batch %d/%d][Loss %f][Corr %f][Class %f\%f]' % (run, runs, epoch, mtlcorr_epochs, i, len(dloader_ft_mtask), loss, corr, t1_loss, t2_loss))


            # scheduler_s.step()
            # scheduler_t.step()
        # check for overfit/underfit

        for handle in handles:
            handle.remove()

        ft_acc_t1.append(evaluate(snet, dloader_ft_t1, device=device))
        ft_acc_t2.append(evaluate(tnet, dloader_ft_t2, device=device))


        running_test_acc_t1.append(evaluate(snet, dloader_test_t1, device=device))
        running_test_acc_t2.append(evaluate(tnet, dloader_test_t2, device=device))
        snet.apply(register_hooks)
        tnet.apply(register_hooks)


    mtl_corr_run_acc_t1.append(running_test_acc_t1)
    mtl_corr_run_acc_t2.append(running_test_acc_t2)


plt.plot(ft_acc_t1, alpha=0.3, color='g', label='ft')
plt.plot(running_test_acc_t1, alpha=0.3, color='r', label='test')
plt.legend()
plt.savefig('t1_ft.png')

plt.close()
plt.plot(ft_acc_t2, alpha=0.3, color='g', label='ft')
plt.plot(running_test_acc_t2, alpha=0.3, color='r', label='test')
plt.legend()
plt.savefig('t2_ft.png')

print('completed')
raise SystemError(0)

##################################################################
# simple mtl
##################################################################
mtl_run_acc_t1 = []
mtl_run_acc_t2 = []
for run in range(runs):

    snet = initilize_resNet(out_classes=2, pretrained=False, load="models/STL_t1_150epochs.ckp", device=device)
    tnet = initilize_resNet(out_classes=2, pretrained=False, load="models/STL_t2_150epochs.ckp", device=device)
    opt_s = torch.optim.Adam(snet.parameters(), lr=0.001, betas=(0.9, 0.999))
    opt_t = torch.optim.Adam(tnet.parameters(), lr=0.001, betas=(0.9, 0.999))


    running_mtl_acc_t1 = []
    running_mtl_acc_t1.append(evaluate(snet, dloader_test_t1, device=device))

    running_mtl_acc_t2 = []
    running_mtl_acc_t2.append(evaluate(tnet, dloader_test_t2, device=device))

    for param in snet.parameters():
        param.requires_grad = False
    snet.apply(bn_requires_grad)
    for param in snet.fc.parameters():
        param.requires_grad = True

    for param in tnet.parameters():
        param.requires_grad = False
    tnet.apply(bn_requires_grad)
    for param in tnet.fc.parameters():
        param.requires_grad = True
    
    for epoch in range (mtlcorr_epochs):
        for i, batch in enumerate(dloader_ft_mtask):

            opt_s.zero_grad()
            opt_t.zero_grad()

            samples = batch[0].to(device)
            labels = batch[1].to(device)

            pred = snet(samples)
            t1_loss = t1_criterion(pred, labels[:, 0])


            pred = tnet(samples)
            t2_loss = t2_criterion(pred, labels[:, 1])

            loss = t1_loss + t2_loss
            loss.backward()

            # update parameters
            opt_s.step()
            opt_t.step()

            print('[MTL][Run %d/%d][Epoch %d/%d][Batch %d/%d][Loss %f]' % (run, runs,epoch, mtlcorr_epochs, i, len(dloader_ft_t2), loss))
        running_mtl_acc_t1.append(evaluate(snet, dloader_test_t1, device=device))
        running_mtl_acc_t2.append(evaluate(tnet, dloader_test_t2, device=device))

    mtl_run_acc_t1.append(running_mtl_acc_t1)
    mtl_run_acc_t2.append(running_mtl_acc_t2)
# print("Saving MTL")
# torch.save(snet.state_dict(), "MTL_source.ckp")
# torch.save(tnet.state_dict(), "MTL_target.ckp")

print('Statistics and plotting')
print('mtl-corr base %f, %f ' %(running_test_acc_t1[0], running_test_acc_t2[0]))
print('mtl base %f, %f' %(running_mtl_acc_t1[0], running_mtl_acc_t2[0] ))

for run in range(runs):
    plt.plot(mtl_corr_run_acc_t1[run], alpha=0.3, color='g')

for run in range(runs):
    plt.plot(mtl_run_acc_t1[run], alpha=0.3, color='r')

plt.plot(running_ft_acc_t1, alpha=0.3, color='r')

plt.savefig('t1_test.png')
plt.close()

for run in range(runs):
    plt.plot(mtl_corr_run_acc_t2[run], alpha=0.3, color='g')

for run in range(runs):
    plt.plot(mtl_run_acc_t2[run], alpha=0.3, color='r')

plt.plot(running_ft_acc_t2, alpha=0.3, color='r')

plt.savefig('t2_test.png')




raise SystemError(0)
for handle in handles:
    handle.remove()

# evaluate mtask accuracy
# mtl_corr_acc = evaluate([snet, tnet], dloader_test_mtask)
mtl_corr_acc = []
mtl_corr_acc.append(evaluate(snet, dloader_test_t1))
mtl_corr_acc.append(evaluate(tnet, dloader_test_t2))
print("base acc: ", base_acc)
print("mtl acc: ", mtl_corr_acc)

##################################################################
# MTL MODEL
##################################################################

mtl_model = resnet18(pretrained=False, progress=True)
num_ftrs = mtl_model.fc.in_features
# for i, model in enumerate(list(mtl_model.children())[:-1]):
#     print(i, model)

mtl_model = nn.Sequential(*list(mtl_model.children())[:-1])

t1_classifier = nn.Linear(num_ftrs, 1)
t2_classifier = nn.Linear(num_ftrs, 1)

mtl_model = mtl_model.to(device)
t1_classifier = t1_classifier.to(device)
t2_classifier = t2_classifier.to(device)

opt_mtl = torch.optim.Adam(list(mtl_model.parameters()) + list(t1_classifier.parameters()) + list(t2_classifier.parameters()), lr=0.001, betas=(0.9, 0.999))
# opt_mtl  =torch.optim.RMSprop(list(mtl_model.parameters()) + list(t1_classifier.parameters()), lr=0.045, alpha=0.98, eps=1e-08, weight_decay=0, momentum=0, centered=False)

d_loader_train_ft = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([train_mtask, ft_mtask]), batch_size=128, shuffle=True, num_workers=2)
for epoch in range(mtl_epochs):
    for i, batch in enumerate(d_loader_train_ft):

        opt_mtl.zero_grad()

        
        samples = batch[0].to(device)
        labels = batch[1].to(device)


        features = mtl_model(samples)
        features = features.reshape(features.shape[0], -1, num_ftrs)
        # print(features.shape)

        t1_pred = t1_classifier(features).squeeze()
        t1_loss = t1_criterion(t1_pred, labels[:, 0].float())

        t2_pred = t2_classifier(features).squeeze()
        t2_loss = t2_criterion(t2_pred, labels[:, 1].float())

        loss = t1_loss + t2_loss

        # backprop
        loss.backward()

        opt_mtl.step()

        print('[MTL][Epoch %d/%d][Batch %d/%d][Loss %f]' % (epoch, mtl_epochs, i, len(d_loader_train_ft), loss))

s_correct = 0
t_correct = 0
total = 0


mtl_model.eval()
for i, batch in enumerate(dloader_test_mtask):

    samples = batch[0].to(device)
    labels = batch[1].to(device)

    features = mtl_model(samples)
    features = features.reshape(features.shape[0], -1, num_ftrs)

    t1_pred = t1_classifier(features).squeeze()
    t1_pred = torch.sigmoid(t1_pred)
    t1_pred = torch.where(t1_pred>0.5, torch.ones_like(t1_pred), torch.zeros_like(t1_pred)).squeeze()
    s_correct += (t1_pred == labels[:, 0]).sum().item()


    t2_pred = t2_classifier(features).squeeze()
    t2_pred = torch.sigmoid(t2_pred)
    t2_pred = torch.where(t2_pred>0.5, torch.ones_like(t2_pred), torch.zeros_like(t2_pred)).squeeze()
    t_correct += (t2_pred == labels[:, 1]).sum().item()

    total += labels.shape[0]

mtl_accuracy_s = s_correct / total
mtl_accuracy_t = t_correct / total



# print("**********STL**********")
# print("STL-t1-no-ft:", stl_acc_train_t1)
# print("STL-t2-no-ft:", stl_acc_train_t2)
# print("STL-t1-ft:", stl_acc_ft_t1)
# print("STL-t2-ft:", stl_acc_ft_t2)

# print("**********MTLCorr**********")
# print("MTLcorr-t1-ft:", mtl_corr_acc[0])
# print("MTLcorr-t2-ft:", mtl_corr_acc[1])


# print("**********MTL**********")
# print("MTL-t1-ft:", mtl_accuracy_s)
# print("MTL-t2-ft:", mtl_accuracy_t)
# print('Completed')

# print(running_train_acc_t1)

# print("train_t1: ", running_train_acc_t1)
# print("train_t2: ", running_train_acc_t2)
# print("test_t1: ", running_test_acc_t1)
# print("test_t2: ", running_test_acc_t2)

# snet.eval()
# tnet.eval()
# for i, batch in enumerate(dloader_test_mtask):

#     samples = batch[0].to(device)
#     labels = batch[1].to(device)
    
#     s_pred = snet(samples)
#     _, pred = torch.max(s_pred.data, 1)
#     s_correct += (pred == labels[:, 0]).sum().item()


#     t_pred = tnet(samples)
#     _, pred = torch.max(t_pred.data, 1)
#     print(pred)
#     t_correct += (pred == labels[:, 1]).sum().item()

#     total += labels.shape[0]

# mtl_corr_accuracy_s = s_correct / total
# mtl_corr_accuracy_t = t_correct / total
