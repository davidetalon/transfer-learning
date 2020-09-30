import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet101
from torch.nn.functional import relu
import math
import numpy as np

torch.manual_seed(5)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Intermediate:
    def __init__(self):
        self.tensors=[]
        self.handles=[]
    
    def get_intermediate(self):
        def hook(model, input, output):
            self.tensors.append(output)
        return hook

    def register_hooks(self, model, typeclass):
        for m in model.modules():
            if isinstance(m, typeclass):
                self.handles.append(m.register_forward_hook(self.get_intermediate()))
                # m.eval()

    def remove_handles(self):
        self.tensors.clear()
        for handle in self.handles:
            handle.remove()
        self.handles=[]
# class Attention(nn.Module):

#     def __init__(self, input_dim, embedd_dim):
#         super(Attention, self).__init__()
#         self.embedd_dim = embedd_dim
#         self.q_weights = torch.nn.Parameter(torch.Tensor(input_dim, embedd_dim))
#         self.k_weights = torch.nn.Parameter(torch.Tensor(input_dim, embedd_dim))

#     def forward(self, x, v):
#         print('input:', x.shape, self.q_weights.shape)
#         query = torch.matmul(x, self.q_weights)
#         print('query: ', query.shape)
#         key = torch.matmul(x, self.k_weights)
#         print('key: ', key.shape)

#         attention_weights = torch.matmul(query, key.transpose(1, -1))/math.sqrt(self.embedd_dim)
#         print('correspondence: ',attention_weights.shape)
#         attention_weights = torch.nn.functional.softmax(attention_weights)

#         attended = torch.dot(attention_weights, v)

#         return attended


# class SoftHistogram(nn.Module):
#     def __init__(self, bins, min, max, sigma):
#         super(SoftHistogram, self).__init__()
#         self.bins = bins
#         self.min = min
#         self.max = max
#         self.sigma = sigma
#         self.delta = float(max - min) / float(bins)
#         self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)

class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = torch.nn.Parameter(torch.Tensor([sigma]))
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device=device).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        # x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        return x

class SoftHist(nn.Module):
    def __init__(self, channels, height, width, bins=16, min_value=-0.2, max_value=10, sigma=100, momentum=0.1):
        super(SoftHist, self).__init__()
        self.channels=channels
        self.height=height
        self.width=width
        self.bins = bins
        self.min = min_value
        self.max = max_value
        self.sigma = sigma
        self.momentum = momentum
        self.delta = float(self.max - self.min) / self.bins
        self.centers = float(self.min) + self.delta * (torch.arange(self.bins, device=device).float() + 0.5)

        zeros = torch.zeros((self.channels, self.height, self.width, self.bins), device=device)
        self.register_buffer('running_hist', zeros)

    def forward(self, in_tensor):

        hists = torch.empty((self.channels, self.height, self.width, self.bins), device=device)
        for ch in range(self.channels):
            for i in range(self.height):
                for j in range(self.width):

                    x = in_tensor[:, ch, i, j].reshape(-1)
        
                    x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
                    x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
                    batch_hist = x.sum(dim=1)

                    current_hist = (1-self.momentum) * self.running_hist[ch, i, j].detach() + self.momentum * batch_hist
                    self.running_hist[ch, i, j] = current_hist.detach()

                    # normalize
                    hist = torch.add(current_hist, 1)
                    hist = torch.true_divide(hist, hist.sum())

                    hists[ch, i, j] = hist

        return hists
            

def soft_hist(x, bins, min, max, sigma):
    min_value = min
    max_value = max
    delta = float(max_value - min_value) /float(bins)
    centers =  float(min_value) + delta * (torch.arange(bins, device=device).float() + 0.5)

    x = torch.unsqueeze(x, 0) - torch.unsqueeze(centers, 1)
    x = torch.sigmoid(sigma * (x + delta/2)) - torch.sigmoid(sigma * (x - delta/2))
    x = x.sum(dim=1)

    return x

class BaseModel(nn.Module):
    def __init__(self, n_classes):
        super(BaseModel, self).__init__()
        self.flatten = nn.Flatten()
        self.in_layer = nn.Linear(12288, 64)
        self.layer1 = nn.Sequential(
                        nn.Linear(64, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        FilmLayer(64, channels_last=True))

        self.layer2 = nn.Sequential(
                        nn.Linear(64, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        FilmLayer(64))

        self.layer3 = nn.Sequential(
                        nn.Linear(64, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        FilmLayer(64))

        self.layer4 = nn.Sequential(
                        nn.Linear(64, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        FilmLayer(64))
    
        self.fc = nn.Linear(64, 2)
    
    def forward(self, x):

        x = self.flatten(x)

        x = self.in_layer(x)
        
        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(x)
        return x
    def clip_film_layers(self):
        for m in self.modules():
            if isinstance(m, FilmLayer):
                # identity transformation
                m.identity()

                # do not update, do not compute the gradient
                for param in m.parameters():
                    param.requires_grad=False

    def unclip_film_layers(self):
        for m in self.modules():
            if isinstance(m, FilmLayer):
                # initialize
                m.initialize_params()

                # compute gradient
                for param in m.parameters():
                    param.requires_grad=True

class AugmentedResnet(nn.Module):

    def __init__(self, n_classes, film_layers=4):
        super(AugmentedResnet, self).__init__()
        
        # initialize a resnetmodel with film layers(Id transformation)
        # unlock film layers to be trained
        resnet = resnet18(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        layer1 = []
        layer1.append(resnet.layer1[0])
        layer1.append(FilmLayer(resnet.layer1[0].bn2.num_features))
        layer1.append(resnet.layer1[1])
        layer1.append(FilmLayer(resnet.layer1[1].bn2.num_features))
        self.layer1 = nn.Sequential(*layer1)

        layer2 = []
        layer2.append(resnet.layer2[0])
        layer2.append(FilmLayer(resnet.layer2[0].bn2.num_features))
        layer2.append(resnet.layer2[1])
        layer2.append(FilmLayer(resnet.layer2[1].bn2.num_features))
        self.layer2 = nn.Sequential(*layer2)     

        layer3 = []
        layer3.append(resnet.layer3[0])
        layer3.append(FilmLayer(resnet.layer3[0].bn2.num_features))
        layer3.append(resnet.layer3[1])
        layer3.append(FilmLayer(resnet.layer3[1].bn2.num_features))
        self.layer3 = nn.Sequential(*layer3)  

        layer4 = []
        layer4.append(resnet.layer4[0])
        layer4.append(FilmLayer(resnet.layer4[0].bn2.num_features))
        layer4.append(resnet.layer4[1])
        layer4.append(FilmLayer(resnet.layer4[1].bn2.num_features))
        self.layer4 = nn.Sequential(*layer4) 

        # self.layer4 = nn.Sequential(*list(resnet.layer4))
        # self.film4 = FilmLayer(self.layer4[-1].bn2.num_features)
  
        self.avgpool = resnet.avgpool

        num_ftrs = resnet.fc.in_features
        self.fc = nn.Linear(num_ftrs, n_classes)

    def clip_film_layers(self):
        for m in self.modules():
            if isinstance(m, FilmLayer):
                # identity transformation
                m.identity()

                # do not update, do not compute the gradient
                for param in m.parameters():
                    param.requires_grad=False

    def unclip_film_layers(self):
        for m in self.modules():
            if isinstance(m, FilmLayer):
                # initialize
                # m.initialize_params()

                # compute gradient
                for param in m.parameters():
                    param.requires_grad=True


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.film1(x)

        x = self.layer2(x)
        # x = self.film2(x)

        x = self.layer3(x)
        # x = self.film3(x)

        x = self.layer4(x)
        # x = self.film4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class FilmLayer(nn.Module):
    def __init__(self, in_channels, channels_last=False):

        super(FilmLayer, self).__init__()
        self.in_channels = in_channels
        self.channels_last = channels_last
        self.gamma = torch.nn.Parameter(torch.empty(in_channels, 1, 1))
        self.beta = torch.nn.Parameter(torch.empty(in_channels, 1, 1))

        self.initialize_params()

    def forward(self, x):
        if self.channels_last:
            x = x.transpose(1, -1)

        # scaling
        x = x * self.gamma
        # biasing
        x = x + self.beta

        if self.channels_last:
            x = x.transpose(1, -1)

        return x
    def initialize_params(self):
        torch.nn.init.kaiming_normal_(self.gamma, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.beta, a=math.sqrt(5))
    
    def identity(self):
        torch.nn.init.constant_(self.gamma, 1)
        torch.nn.init.constant_(self.beta, 0)


class ShallowLinear(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_units = 512):
        super(ShallowLinear, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels


        self.in_layer = nn.Linear(3, hidden_units)
        self.film_in = FilmLayer(hidden_units, channels_last=True)

        self.layer1 = nn.Linear(hidden_units, hidden_units//2)
        self.film1 = FilmLayer(hidden_units//2, channels_last=True)

        self.layer2 = nn.Linear(hidden_units//2, hidden_units//4)
        self.film2 = FilmLayer(hidden_units//4, channels_last=True)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(524288, out_channels)
    
    
    def forward(self, x):

        x = torch.transpose(x, 1, -1)

        x = self.in_layer(x)
        x = relu(x)
        x = self.film_in(x)

        x = self.layer1(x)
    
        x = relu(x)
        x = self.film1(x)

        x = self.layer2(x)
        x = relu(x)
        x  =self.film2(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = torch.transpose(x, -1, 1)

        return x

    def clip_film_layers(self):
        for m in self.modules():
            if isinstance(m, FilmLayer):
                # identity transformation
                m.identity()

                # do not update, do not compute the gradient
                for param in m.parameters():
                    param.requires_grad=False

    def unclip_film_layers(self):
        for m in self.modules():
            if isinstance(m, FilmLayer):
                # initialize
                m.initialize_params()

                # compute gradient
                for param in m.parameters():
                    param.requires_grad=True

                                                                  

if __name__ == '__main__':
    
    import torch
    import os
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.datasets import FakeData
    from torch.utils.data import DataLoader
    from torchvision.models.resnet import BasicBlock
    from torchvision.transforms import ToTensor

    data = FakeData(size =1000, image_size=(3,32,32), num_classes=4, transform=ToTensor())
    dloader = DataLoader(data, batch_size=32, shuffle=True, num_workers=4)
  
    model = resnet18().to(device)
    
    optim = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

    mse = nn.MSELoss().to(device)

    hooks_manager = Intermediate()
    

    hooks_manager.register_hooks(model, BasicBlock)
    out = model(iter(dloader).next()[0].to(device))
    targets = []
    hist_layers = []
    for i, tens in enumerate(hooks_manager.tensors):

        targets.append(F.softmax(torch.randint(low=0, high=125, size=(tens.shape[1],16), dtype=torch.float, device =device), dim=0))
        hist_layers.append(SoftHist(tens.shape[1], bins=8, min_value=0, max_value=10, sigma=100, momentum=0.3))

    hooks_manager.remove_handles()
    
    n_epochs = 30
    for epoch in range(n_epochs):
        for i, batch in enumerate(dloader):
            
            hooks_manager.register_hooks(model, BasicBlock)
            optim.zero_grad()

            samples=batch[0].to(device)

            out = model(samples)


            loss = 0
            for lay, tensor in enumerate(hooks_manager.tensors):
                computed_hist =hist_layers[lay](tensor)
                print(computed_hist[0])
                # print(targets[lay][0])
                mse_loss = mse(computed_hist, targets[lay])
                print(mse_loss)
                loss = loss + mse_loss



            loss.backward()
            optim.step()
            
            hooks_manager.remove_handles()

            print('Epoch %d/%d][Batch %d/%d][Loss %d]' % (epoch, n_epochs, i, len(dloader), loss))

