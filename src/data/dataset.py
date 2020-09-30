import torchvision
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os, sys
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(6)

def load_image(path):
    with open(path, 'rb') as f:
        image = Image.open(f)
        return image.convert('RGB')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
class MultitaskLabels(object):
    def __init__(self, mantained):
        self.mantained = mantained
    
    def __call__(self, label):

        if len(self.mantained)==1:

            labels = 1 if label in self.mantained[0] else 0
        else:
            labels = []
            
            for task in self.mantained:
                labels.append(1 if label in task else 0)
            labels = np.array(labels)

        return labels

class SoftLabels(object):
    def __init__(self, model):
        self.model = model
    
    def __call__(self, sample):

        if isinstance(self.model, torch.nn.modules.Module):
            label = self.module(sample)
        else:
            label = []
            for model in self.model:
                label.append(model(sample))
        return label

class MNISTDataset(Dataset):

    def __init__(self, root, classes=[6, 8], image_set='train', transform=None, target_transform=None):
        
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        if image_set=='train':
            data_file = 'training.pt'
        elif image_set == 'ft':
            data_file = 'ft.pt'
        else:
            data_file = 'test.pt'
        data, targets = torch.load(os.path.join(self.root, data_file))

        self.data = data[torch.bitwise_or(targets==classes[0], targets ==classes[1]), :, :]
        self.targets = targets[torch.bitwise_or(targets==classes[0], targets ==classes[1])]
        self.targets[self.targets==classes[0]] = 0
        self.targets[self.targets==classes[1]] = 1
        

    def __getitem__(self, index):

        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class BinaryCIFAR(Dataset):

    def __init__(self, root, tasks=[(0, 1)], transform=None, target_transform=None):
        self.n_tasks = len(tasks)
        self.tasks= tasks
        
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.requires_soft = False
        self.soft_labels = []
        data_dir = Path(self.root)

        
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        with open(data_dir, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])

        

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.targets = np.array(self.targets)

        self.data = np.array(self.data).reshape(-1, 3, 32, 32)

        # select a small batch - overfit to it

        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # selected = np.bitwise_or(self.targets==c1, self.targets == c2)
        # self.data = self.data[selected]
        
        # self.targets = self.targets[selected]
        # self.targets[self.targets==c1] = 0
        # self.targets[self.targets==c2] = 1
        
        
        tasks_data = []
        tasks_targets =[]
        for task in tasks:
            imgs = []
            classes = []
            for idx_cl, cl in enumerate(task):
                idxs = (self.targets==cl)
                selected = self.data[idxs]
                # print(len(selected))
                imgs.extend(selected)
                classes.extend([idx_cl for i in range(len(selected))])
                # print(classes)

            mapIndexPosition = list(zip(imgs, classes))
            random.shuffle(mapIndexPosition)
            imgs, classes = zip(*mapIndexPosition)
            tasks_data.append(imgs)
            tasks_targets.append(classes)

        

        self.data = tasks_data
        self.targets = tasks_targets






    def __getitem__(self, index):
        
        
        idx = index % len(self.data[0])

        samples = [self.data[task][idx] for task in range(self.n_tasks)]
        # samples = [load_image(sample_path) for sample_path in sample_data]

        samples = [Image.fromarray(img) for img in samples]
        if self.transform:
            samples = [self.transform(sample) for sample in samples]


        targets = [self.targets[task][idx] for task in range(self.n_tasks)]
        if self.target_transform:
            targets = [self.target_transform(target) for target in targets]

        if self.requires_soft:
            if self.n_tasks == 1:
                return samples[0], targets[0], self.soft_labels[0][idx]
            else:
                return [(samples[task], targets[task], self.soft_labels[task][idx]) for task in range(self.n_tasks)]

        if self.n_tasks == 1:
            return samples[0], targets[0]
        else:
            return [(samples[task], targets[task]) for task in range(self.n_tasks)]


    def __len__(self):
        return len(self.data[0])

    def produce_soft_labels(self, models, batch_size, device):
        soft_labels = []
        for task in range(self.n_tasks):
            # batch data
            data_batches = np.array_split(self.data[task], len(self.data[0])//batch_size + 1, axis=0)
            labels = []
            for idx_batch, batch in enumerate(data_batches):
                
                # prepare data
                tensor_batch = []
                for sample in batch:
                    transformed = Image.fromarray(sample)
                    if self.transform:
                        transformed = self.transform(transformed)

                    tensor_batch.append(transformed)

                tensor_batch = list(map(lambda x:x.unsqueeze(0), tensor_batch))
                tensor_batch = torch.cat(tensor_batch, dim = 0).to(device)

                lab = models[task](tensor_batch).detach().cpu().numpy()

                if idx_batch==0:
                    labels = lab
                else:
                    labels = np.append(labels, lab, axis=0)
            soft_labels.append(labels)
        # soft labels are available
        self.requires_soft=True
        self.soft_labels = soft_labels

class CIFARDataset(Dataset):

    def __init__(self, root, transform=None, target_transform=None, static_target_transform=None):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.requires_soft = False
        self.soft_labels = []
        data_dir = Path(self.root)
        
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        with open(data_dir, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])

        

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = np.array(self.data).reshape(-1, 3, 32, 32)

        # select a small batch - overfit to it

        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC



    def __getitem__(self, index):

        idx = index % len(self.data)

        img, target = self.data[idx], self.targets[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.requires_soft:
            soft_labels = []
            for task in range(len(self.soft_labels)):
                soft_labels.append(self.soft_labels[task][idx])
            return img, target, soft_labels

        return img, target


    def __len__(self):
        return len(self.data)

    def produce_soft_labels(self, models, batch_size, device):

        # batch data
        data_batches = np.array_split(self.data, self.data.shape[0]//batch_size + 1, axis=0)
        labels = []
        for idx_batch, batch in enumerate(data_batches):
            
            # prepare data
            tensor_batch = []
            for sample in batch:
                transformed = Image.fromarray(sample)
                if self.transform:
                    transformed = self.transform(transformed)

                tensor_batch.append(transformed)

            tensor_batch = list(map(lambda x:x.unsqueeze(0), tensor_batch))
            tensor_batch = torch.cat(tensor_batch, dim = 0).to(device)

            mt_labels = []
            for model in models:
                mt_labels.append(model(tensor_batch).detach().cpu().numpy())

            if idx_batch==0:
                labels.extend(mt_labels)
            else:
                for i, mt_label in enumerate(mt_labels):
                    labels[i] = np.append(labels[i], mt_label, axis=0)
        
        # soft labels are available
        self.requires_soft=True
        self.soft_labels = labels

class ImageDataset(Dataset):

    def __init__(self, root, transform=None, target_transform=None):
       
        self.root = root
        self.transform = transform
        self.target_transform = transform

        # get all files, iterate over all classes
        data_dir = Path(self.root)

        classes = os.listdir(data_dir)
            
        image_list=[]
        for cl in classes:

            
            image_list.extend([(f, cl) for f in (data_dir / cl).iterdir()])
        

        self.image_list = image_list
        self.classes = classes

        # gather minimum dimension
        print('%d images divided into %d classes' % (len(self.image_list), len(classes)))

    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        
        image_path, cl = self.image_list[idx]

        # load image
        image = load_image(image_path)
        label = self.classes.index(cl)

        # apply transformation
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(label)

        return image, target


class PascalVOC2012(Dataset):
    def __init__(self, root, task='classification', image_set='train',transform=None, target_transfor=None):

        self.transform = transform
        self.target_transform = transform

        voc_root = Path(root)

        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')  
        seg_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        ###################
        # Annotaions
        ###################
        annot_split_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(annot_split_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        # get annotations
        annotation_dir = os.path.join(voc_root, 'Annotations')
        images_annot = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]

        ###################
        # Segmentation masks
        ###################
        split_f = os.path.join(seg_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        images_seg = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

        # get paired data
        annot_dict = dict((k, i) for i, k in enumerate(images_annot))
        inter = set(annot_dict).intersection(images_seg)
        annot_idxs = [ annot_dict[x] for x in inter ]

        seg_dict = dict((k, i) for i, k in enumerate(images_seg))
        inter = set(seg_dict).intersection(images_annot)
        seg_idxs = [ seg_dict[x] for x in inter ]
        assert (len(seg_idxs) == len(annot_idxs))

        data = []
        labels =[]
        for img in range(len(annot_idxs)):
            data.append(images_annot[annot_idxs[img]])
            labels.append({'annot': annotations[annot_idxs[img]],
                        'mask': masks[seg_idxs[img]]})

        self.images = data
        self.labels = labels


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        idx = idx % len(self.images)

        sample = self.images[idx]
        target = self.labels[idx]

        return sample, target
    


class Places365(Dataset):
    def __init__(self, root, image_set='train', tasks=[(0, 1)], transform=None, target_transform=None):
        self.n_tasks = len(tasks)
        self.tasks = tasks
        self.transform = transform
        self.target_transform = target_transform
        self.requires_soft = False
        self.soft_labels = []
        
        root = Path(root)
        # if image_set == 'train':
        #     ground_file = 'places365_train_standard.txt'
        # else:
        #     ground_file = 'places365_' + image_set + '.txt'

        ground_file = 'places365_' + image_set + '.txt'
        data_dir = os.path.join(root, image_set + "_256")
        
        with open(os.path.join(root, ground_file), 'r') as f:
            paths = []
            classes = []
            for line in f.readlines():
                path, cl = line.strip().split()
                # if image_set == 'train':
                paths.append(os.path.join(data_dir, path))
                # else:
                    # paths.append(os.path.join(data_dir, path))
                classes.append(int(cl))

        self.paths = paths
        self.classes = classes

        self.paths = np.array(self.paths)
        self.classes = np.array(self.classes)

        tasks_data = []
        tasks_targets =[]
        for task in tasks:
            paths = []
            classes = []
            for idx_cl, cl in enumerate(task):
                selected = self.paths[self.classes==cl]
                paths.extend(self.paths[self.classes==cl])
                classes.extend([idx_cl for i in range(len(selected))])
            tasks_data.append(paths)
            tasks_targets.append(classes)

        self.paths = tasks_data
        self.classes = tasks_targets


        #     raise SystemError(0)    
        #     task_idxs = np.logical_or.reduce(selected_idxs)
        
        # # print(len(self.classes))
        # # print(len(self.classes==0))

        # c1, c2 = 0, 1
        # selected = np.bitwise_or(self.classes==c1, self.classes == c2)
        # self.paths = self.paths[selected]
        
        # self.classes = self.classes[selected]
        # self.classes[self.classes==c1] = 0
        # self.classes[self.classes==c2] = 1

        # assert len(paths) == len (classes)


        # print(filenames[:], 1])
    
    def __len__(self):
        return len(self.paths[0])
    
    def __getitem__(self, idx):
        
        idx = idx % len(self.paths[0])

        sample_paths = [self.paths[task][idx] for task in range(self.n_tasks)]
        samples = [load_image(sample_path) for sample_path in sample_paths]


        if self.transform:
            samples = [self.transform(sample) for sample in samples]


        targets = [self.classes[task][idx] for task in range(self.n_tasks)]
        if self.target_transform:
            targets = [self.target_transform(target) for target in targets]

        if self.requires_soft:
            if self.n_tasks == 1:
                return samples[0], targets[0], self.soft_labels[0][idx]
            else:
                return [(samples[task], targets[task], self.soft_labels[task][idx]) for task in range(self.n_tasks)]

        if self.n_tasks == 1:
            return samples[0], targets[0]
        else:
            return [(samples[task], targets[task]) for task in range(self.n_tasks)]


    def produce_soft_labels(self, models, batch_size, device):
        soft_labels = []
        for task in range(self.n_tasks):
            # batch data
            data_batches = np.array_split(self.paths[task], len(self.paths[0])//batch_size + 1, axis=0)
            labels = []
            for idx_batch, batch in enumerate(data_batches):
                
                # prepare data
                tensor_batch = []
                for sample in batch:
                    transformed = load_image(sample)
                    if self.transform:
                        transformed = self.transform(transformed)

                    tensor_batch.append(transformed)

                tensor_batch = list(map(lambda x:x.unsqueeze(0), tensor_batch))
                tensor_batch = torch.cat(tensor_batch, dim = 0).to(device)

                lab = models[task](tensor_batch).detach().cpu().numpy()
                if idx_batch==0:
                    labels = lab
                else:
                    labels = np.append(labels, lab, axis=0)

            soft_labels.append(labels)

        # soft labels are available
        self.requires_soft=True
        self.soft_labels = soft_labels
    


if __name__=='__main__':

    import numpy as np
    from torchvision.transforms import ToTensor, Compose, Resize, Normalize
    from torchvision.models import resnet34, resnet18, alexnet, mobilenet_v2
    from torchvision.datasets import VOCSegmentation

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # train_data = Places365('data/processed/mini-places365_2', image_set='ft', tasks=[(3, 4), (5, 6), (7,8)], transform=transform)
    # dloader = DataLoader(train_data, batch_size=5,  shuffle=True, num_workers=4)

    # model = resnet18()

    # for i, batch in enumerate(dloader):

    #     print(len(batch))
    #     out = model(batch[0][0])
    #     print(out.shape)

    #     if(i==0):
    #         break

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_data = BinaryCIFAR(root='data/processed/cifar/train', tasks=[(3,4), (5, 6)], transform=transform)
    dloader = DataLoader(train_data, batch_size=32,  shuffle=True, num_workers=4)

    models=[resnet18().to(device), resnet18().to(device)]
    dloader.dataset.produce_soft_labels(models, 128, device)

    n_tasks = 2
    n_epochs = 50

    optims = [  torch.optim.SGD(models[0].parameters(), lr=0.001, momentum=0.9),
                torch.optim.SGD(models[1].parameters(), lr=0.001, momentum=0.9)]
    ce = torch.nn.CrossEntropyLoss().to(device)

    for task in range(n_tasks):
        for epoch in range(n_epochs):
            for i, batch in enumerate(dloader):
                
                optims[task].zero_grad()

                sample = batch[task][0].to(device)
                label = batch[task][1].to(device)

                pred = models[task](sample)


                loss = ce(pred, label)

                loss.backward()
                optims[task].step()

                print('[Taks %d/%d][Epoch %d/%d][Batch %d/%d][Loss %f]' % (task, n_tasks, epoch, n_epochs, i, len(dloader), loss))



    # voc_detection = VOCSegmentation('data/raw/VOC', year='2012', image_set='train', download=False, transform=None, target_transform=None, transforms=transform)
    # dloader = DataLoader(voc_detection, batch_size=1, shuffle=True, num_workers=4)

    # for i, batch in enumerate(dloader):
    #     print(batch)

    #     if(i==0):
    #         break

    # train = PascalVOC2012('data/raw/VOC2012', image_set='train', transform=None, target_transfor=None)
    # dloader = DataLoader(train, batch_size=2, shuffle=True, num_workers=4)
    # for i, batch in enumerate(dloader):
    #     print(batch)

    #     if(i==0):
    #         break