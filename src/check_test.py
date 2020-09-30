import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, wide_resnet101_2
from torchvision.datasets import FakeData, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from model.baseline import ShallowLinear
from data.dataset import MultitaskLabels
from matplotlib import pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

epochs = 200

def evaluate(model, dloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        size = 0
        for i, batch in enumerate(dloader):
            sample = batch[0].to(device)
            target = batch[1].to(device)

            out = model(sample)
            _, pred = torch.max(out, dim=1)

            correct += (pred==target).float().sum().item()
            size += target.size(0)
    model.train()
    return (correct / size)

task1 = MultitaskLabels([range(0,50,1)])
transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_train = CIFAR100(root='data/raw/cif', train=True, download=False, transform=transform, target_transform=task1)
dloader_train = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=4)

data_test = CIFAR100(root='data/raw/cif', train=False, download=False, transform=transform, target_transform=task1)
dloader_test = DataLoader(data_test, batch_size=128, shuffle=False, num_workers=4)

# training shallow model
# model = wide_resnet101_2()
model = resnet18()
fc_in = model.fc.in_features
model.fc = torch.nn.Linear(fc_in, 2)
model = model.to(device)

# model = ShallowLinear(3,10, 512)
# model = model.to(device)
# model.train()

opt = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, [60, 120, 160], gamma=0.2)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

train_err = []
test_err = []


for epoch in range(epochs):
    for i, batch in enumerate(dloader_train):
    # for i in range(300):

        opt.zero_grad()

        samples = batch[0].to(device)
        labels = batch[1].to(device)

        pred = model(samples)
        loss = loss_fn(pred, labels)

        loss.backward()

        # update parameters
        opt.step()

        print('[Epoch %d/%d][Batch %d/%d][Loss %f]' % (epoch, epochs, i, len(dloader_train), loss))

    sched.step()

    train_err.append(evaluate(model, dloader_train))
    test_err.append(evaluate(model, dloader_test))
    

print("Saving")
torch.save(model.state_dict(), "resnet18CIFAR100.ckp")


plt.plot(train_err, color='k', label='train')
plt.plot(test_err, color='r', label='test')
plt.legend()
plt.savefig('wideresnet.png')
# for epoch in range(32):
#     print(evaluate(model, dloader_test))
