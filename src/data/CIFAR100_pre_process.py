from pathlib import Path
import numpy as np
import sys
import pickle
from shutil import copyfile

source_folder = 'data/raw/cifar100/cifar-100-python/'
target_folder = 'data/processed/cifar/'

train_file = 'train'
test_file = 'test'

source_folder = Path(source_folder)

target_folder = Path(target_folder)
target_folder.mkdir(parents=True, exist_ok=True)


file = source_folder / train_file

# import raw-cifar
data =[]
targets=[]
with open(file, 'rb') as f:
    if sys.version_info[0] == 2:
        entry = pickle.load(f)
    else:
        entry = pickle.load(f, encoding='latin1')
    data.append(entry['data'])
    if 'labels' in entry:
        targets.extend(entry['labels'])
    else:
        targets.extend(entry['fine_labels'])

data = np.array(data).reshape(50000, -1)
targets = np.array(targets)

# choosing the subset
indexes =np.zeros((100, 500), dtype=np.int64)
for label in range(100):
    indexes[label, :] = np.argwhere(targets==label).squeeze()


# fine-tuning
subset = indexes[:, 70:120].reshape(-1)
subset_data = data.take(subset, axis=0)
subset_labels = [int(idx//50) for idx in range(5000)]
subset_labels = np.array(subset_labels)
# dict
subset = {'data':subset_data, 'fine_labels':subset_labels}


# validation
subset_val = indexes[:, :70].reshape(-1)
subset_val_data = data.take(subset_val, axis=0)
subset_val_labels = [int(idx//70) for idx in range(70*100)]
subset_val_labels = np.array(subset_val_labels)
# dict
subset_val = {'data':subset_val_data, 'fine_labels':subset_val_labels}


# training
remaining = indexes[:, 120:].reshape(-1)
remaining_data = data.take(remaining, axis=0)
remaining_labels = [int(idx//380) for idx in range(380*100)]
remaining_labels = np.array(remaining_labels)
# dict
remaining = {'data':remaining_data, 'fine_labels':remaining_labels}


# dumping data
target_file = target_folder / "fine-tune"
with open(target_file, 'wb') as handle:
    pickle.dump(subset, handle, protocol=pickle.HIGHEST_PROTOCOL)

target_file = target_folder / "valid"
with open(target_file, 'wb') as handle:
    pickle.dump(subset_val, handle, protocol=pickle.HIGHEST_PROTOCOL)


target_file = target_folder / train_file
with open(target_file, 'wb') as handle:
    pickle.dump(remaining, handle, protocol=pickle.HIGHEST_PROTOCOL)

# copying remaining
test_source = source_folder / test_file
test_dest = target_folder / test_file
copyfile(test_source, test_dest)


# # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
# self.data = np.array(self.data).reshape(-1, 3, 32, 32)
# self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC