import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
from utils.model import bn_requires_grad, initilize_resNet
from data.dataset import CIFARDataset, MultitaskLabels

torch.manual_seed(5)
# np.random.seed(5)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

mtlcorr_epochs = 20

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

    return correct/ total * 100

transform=ToTensor()
task1 = MultitaskLabels([range(0,50,1)])
test_t1 = CIFARDataset(root='data/processed/cifar/test', transform=transform, target_transform=task1)
dloader_test_t1 = torch.utils.data.DataLoader(test_t1, batch_size=128, shuffle=False, num_workers=4)

task2 = MultitaskLabels([list(range(30, 50, 1)) + list(range(60,90, 1))])
test_t2 = CIFARDataset(root='data/processed/cifar/test', transform=transform, target_transform=task2)
dloader_test_t2 = torch.utils.data.DataLoader(test_t2, batch_size=128, shuffle=False, num_workers=4)

mtl = MultitaskLabels([range(0,50,1), list(range(30, 50, 1)) + list(range(60,90,1))])
ft_mtask = CIFARDataset(root='data/processed/cifar/fine-tune', transform=transform, target_transform=mtl)
dloader_ft_mtask = torch.utils.data.DataLoader(ft_mtask, batch_size=128, shuffle=True, num_workers=4)



models=[]
models.append(initilize_resNet(out_classes=2, pretrained=False, load="models/STL_t1_no_pre_proc.ckp", device=device))
models.append(initilize_resNet(out_classes=2, pretrained=False, load="models/STL_t2_no_pre_proc.ckp", device=device))

opts = []
opts.append(torch.optim.Adam(models[0].parameters(), lr=0.001, betas=(0.9, 0.999)))
opts.append(torch.optim.Adam(models[1].parameters(), lr=0.001, betas=(0.9, 0.999)))

t1_criterion = torch.nn.CrossEntropyLoss().to(device)
t2_criterion = torch.nn.CrossEntropyLoss().to(device)

for model in models:
    for param in model.parameters():
        param.requires_grad = False
    model.apply(bn_requires_grad)
for param in model.fc.parameters():
    param.requires_grad = True


running_mtl_acc_t1 = []
running_mtl_acc_t1.append(evaluate(models[0], dloader_test_t1, device=device))
running_mtl_acc_t2 = []
running_mtl_acc_t2.append(evaluate(models[1], dloader_test_t2, device=device))

for epoch in range (mtlcorr_epochs):
    for i, batch in enumerate(dloader_ft_mtask):

        opts[0].zero_grad()
        opts[1].zero_grad()

        samples = batch[0].to(device)
        labels = batch[1].to(device)

        pred = models[0](samples)
        t1_loss = t1_criterion(pred, labels[:, 0])


        pred = models[1](samples)
        t2_loss = t2_criterion(pred, labels[:, 1])

        loss = t2_loss
        loss.backward()

        # update parameters
        opts[0].step()
        opts[1].step()

        print('[MTL][Epoch %d/%d][Batch %d/%d][Loss %f]' % (epoch, mtlcorr_epochs, i, len(dloader_ft_mtask), loss))
    running_mtl_acc_t1.append(evaluate(models[0], dloader_test_t1, device=device))
    running_mtl_acc_t2.append(evaluate(models[1], dloader_test_t2, device=device))
