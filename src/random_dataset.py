import torch
from torch import nn
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop
from torchvision.datasets import FakeData
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from utils.correlation import max_corr

from data.dataset import Places365
import tqdm



intermediate_tensors = []
def save_tensor(self, input, output):
    # halfspace = torch.split(output, int(output.shape[1]//2),dim=1)
    # intermediate_tensors.append(output)s
    intermediate_tensors.append(output[:, :int(output.shape[1]//2), :, :])

handles = []
def save_intermediate(m):
    if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
        handles.append(m.register_forward_hook(save_tensor))

def train_mtl(batch, models, losses, opt, device):

    opt.zero_grad()

    sample = batch[0].to(device)
    labels = batch[1]

    batch_losses = []
    for idx_mod, model in enumerate(models):
        task_label = labels[idx_mod].to(device)

        out = model(sample)
        batch_losses.append(losses[idx_mod](out, task_label))
        _, pred = torch.max(out, dim=1)
        correct = (task_label==pred).sum().item()


    corr = 0
    for vx in range(20):
        corr = corr + max_corr(intermediate_tensors[vx], intermediate_tensors[20 + vx], device = device)
    
    # loss = 10*torch.sum(torch.stack(batch_losses))-corr/40
    loss = batch_losses[0] + batch_losses[1] - corr/2
    

    intermediate_tensors[0].retain_grad()
    loss.backward()
    print(intermediate_tensors[0].grad.mean())

    # for name, param in models[0].named_parameters():
    #     if param.requires_grad:
    #         print(name, param.grad.mean())

    opt.step()



    return loss, corr

    

class MultiLabel(object):
    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, label):
        
        labels = []
        # labels.append(label)
        for task in self.classes:
            labels.append(1 if label in task else 0)
        return labels

def evaluate_model(models, dloader):
    accuracies = []
    for idx, model in enumerate(models):
        with torch.no_grad():
            
            model.eval()
            correct = 0
            total = 0
            for i, batch in enumerate(dloader):
                sample = batch[0].to(device)
                labels = batch[1][idx].to(device)

                out = model(sample)
                _, pred = torch.max(out, dim=1)
                print(pred)

                correct += (labels==pred).sum().item()
                total += labels.shape[0]

                print('[Evaluation model %d][Batch %d/%d]' % (idx, i, len(dloader)))
            model.train()
        accuracies.append(float(correct/total))

    return accuracies

# def max_corr(source, target):
#     source_repr = source
#     source_repr = source_repr.reshape(source_repr.shape[0], -1)
#     source_repr = source_repr - source_repr.mean(dim=0, keepdim=True)

#     source_cov = torch.sqrt(torch.diag(torch.mm(source_repr.permute(1,0),source_repr)))
#     source_cov = torch.where(source_cov==0, torch.ones_like(source_cov, device=device), source_cov)
#     source_repr = source_repr / source_cov


#     target_repr = target
#     target_repr = target_repr.reshape(target_repr.shape[0], -1)
#     target_repr = target_repr - target_repr.mean(dim=0, keepdim=True)

#     target_cov = torch.sqrt(torch.diag(torch.mm(target_repr.permute(1,0),target_repr)))
#     target_cov = torch.where(target_cov==0, torch.ones_like(target_cov, device=device), target_cov )
#     target_repr = target_repr / target_cov

#     # compute maximum correlation
#     sigma = torch.zeros(source_repr.shape[1], device=device)
#     for idx,row in enumerate(source_repr.split(1)):
#         sigma = sigma + row.squeeze() * target_repr[idx]
#     sigma = sigma * sigma.sign()

#     return sigma.mean()


if __name__=='__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(34)

    training_epochs =0
    mtl_corr_epochs = 50

    num_tasks = 2

    print("Loading dataset")
    target_transform = MultiLabel(classes=[list(range(0, 90)) + list(range(130, 220)), list(range(255,364)) + list(range(98,170))])
    transform = Compose([RandomResizedCrop(64), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    places_data_train = Places365('data/processed/mini-places365', image_set='train', transform=transform, target_transform=target_transform)
    dloader_train = DataLoader(places_data_train, batch_size=128, shuffle=True, num_workers=4)

    places_data_ft = Places365('data/processed/mini-places365', image_set='ft', transform=transform, target_transform=target_transform)
    dloader_ft = DataLoader(places_data_ft, batch_size=8, shuffle=True, num_workers=4)

    places_data_val = Places365('data/processed/mini-places365', image_set='train', transform=transform, target_transform=target_transform)
    dloader_val = DataLoader(places_data_val, batch_size=128, shuffle=True, num_workers=4)


    print("Initializing models")
    models = []
    for task_idx in range(num_tasks):

        resnet = resnet18()
        fc_in = resnet.fc.in_features
        resnet.fc = torch.nn.Linear(fc_in, 2)
        models.append(resnet.to(device))

    losses = []
    weights = [[1.0, 1.0], [1.0, 1.0]]
    # weights =[[1, 23.33], [1, 2.04]]
    weights = torch.tensor(weights)


    optims = []
    for task_idx in range(num_tasks):
        optims.append(torch.optim.SGD(models[task_idx].parameters(), lr=0.001, momentum=0.9))
    for task in range(num_tasks):
        losses.append(CrossEntropyLoss(weights[task]).to(device))

    # training the model
    print("Training models")
    for idx_mod, model in enumerate(models):
        for epoch in range(training_epochs):
            for i, batch in enumerate(dloader_train):

                optims[idx_mod].zero_grad()

                sample = batch[0].to(device)
                labels = batch[1][idx_mod].to(device)

                out = model(sample)
                loss = losses[idx_mod](out, labels)

                loss.backward()
                optims[idx_mod].step()
                print('[Model %d][Epoch %d/%d][Batch %d/%d][Loss %f]' % (idx_mod, epoch, training_epochs, i, len(dloader_train), loss))


    # print("Evaluating the model")
    # base_accuracies = evaluate_model(models, dloader_val)
    # print(base_accuracies)

    # for i, model in enumerate(models):
    #     torch.save(model.state_dict(), "STL_"+str(i)+".ckp")


    # print("Loading the network")
    # for i, model in enumerate(models):
    #     model.load_state_dict(torch.load("models/STL_"+str(i)+"_50.ckp", map_location=device))

    print("Evaluating base model")
    accuracies = evaluate_model(models, dloader_val)
    print(accuracies)


    # let's rock with fine-tuning

    # we need freeze all layers but BN and classifier
    for model in models:
        for param in model.parameters():
            param.requires_grad = False
        
        # unfreeze BN
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
                for param in m.parameters():
                    param.requires_grad = True
        
        # unfreeze classifier
        for param in model.fc.parameters():
            param.requires_grad = True

    # saving intermediate features
    for model in models:
        model.apply(save_intermediate)

    global_params = []
    for model in models:
        global_params += list(model.parameters())

    # global_opt = torch.optim.RMSprop(global_params, lr=0.045)
    # global_opt = torch.optim.Adam(global_params, lr=0.01, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
    # global_opt = torch.optim.Adam(global_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.5, amsgrad=False)
    global_opt = torch.optim.SGD(global_params, lr=0.003, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(global_opt, step_size=3, gamma=0.97)



    print("Start trainig MTL")
    for epoch in range(mtl_corr_epochs):
        for i, batch in enumerate(dloader_ft):

            loss, corr = train_mtl(batch, models, losses, global_opt, device)
            intermediate_tensors = []

            print('[MTLCorr][Epoch %d/%d][Batch %d/%d][Loss %f][Corr %f]' % (epoch, mtl_corr_epochs, i, len(dloader_ft), loss, corr))
        
        scheduler.step()
    
    # for i, model in enumerate(models):
    #     torch.save(model.state_dict(), "MTL_"+str(i)+".ckp")

    # print(base_accuracies)

    for handle in handles:
        handle.remove()

    # accuracies = evaluate_model(models, dloader_val)
    # print(accuracies)