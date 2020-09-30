import torch
from shutil import copyfile
from pathlib import Path

input_file = 'data/raw/MNIST/processed/training.pt'
in_test = 'data/raw/MNIST/processed/test.pt'

out_train = 'data/processed/MNIST/training.pt'
out_ft = 'data/processed/MNIST/ft.pt'
out_test = 'data/processed/MNIST/test.pt'


dst_root = 'data/processed/MNIST'
dst_root = Path(dst_root)
dst_root.mkdir(parents=True, exist_ok=True)

train_data, train_targets = torch.load(input_file)

print(train_data.shape)
print(train_targets.shape)

ft_imgs = []
ft_labels = []
train_imgs = []
train_labels = []
for target in torch.unique(train_targets).tolist():
    ft_imgs.append(train_data[train_targets==target, :, :][:50])
    ft_labels.append(train_targets[train_targets==target][:50])

    train_imgs.append(train_data[train_targets==target, :, ][50:])
    train_labels.append(train_targets[train_targets==target][50:])

ft_data = torch.cat(ft_imgs)
ft_targets = torch.cat(ft_labels)
torch.save([ft_data, ft_targets], out_ft )


train_data = torch.cat(train_imgs)
train_targets = torch.cat(train_labels)
torch.save([train_data, train_targets], out_train )


copyfile(in_test, out_test)
# print(train_targets.shape)