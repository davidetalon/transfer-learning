from pathlib import Path
import os
import numpy as np
from shutil import copyfile
from distutils.dir_util import copy_tree

src_root = 'data/raw/places365'
src_root = Path(src_root)
src_train = os.path.join(src_root, 'train_256')

dst_root = 'data/processed/mini-places365'
dst_root = Path(dst_root)
dst_root.mkdir(parents=True, exist_ok=True)


# read file
paths = []
classes = []
train_file = os.path.join(src_root, "places365_train_standard.txt")
with open(train_file, 'r') as f:
    for line in f.readlines():
        path, cl = line.strip().split()
        paths.append(path[1:])
        classes.append(int(cl))

# building dirs tree on the dst_train folder
print('Building train tree')
dst_train = os.path.join(dst_root, 'train_256')
with open(os.path.join(src_root, 'categories_places365.txt')) as f:
    for cat in f.readlines():
        cat_folder = (cat.strip().split()[0])[1:]
        dst_folder = os.path.join(dst_train, cat_folder)
        os.makedirs(dst_folder, exist_ok=True)

# copying files
print('Copying train files')
indexes = []
classes = np.array(classes)
for label in range(365):
    indexes.append(np.argwhere(classes==label).squeeze()[:2000])
    for img in indexes[label]:
        src = os.path.join(src_train, paths[img])
        dst = os.path.join(dst_train, paths[img])
        copyfile(src, dst)

# writing the classes file
print('Writing train file')
dst_train_file = os.path.join(dst_root, 'places365_train.txt')
with open(dst_train_file, 'w') as f:
    for label in range(365):
        for img in indexes[label]:
            f.write(paths[img]+' '+str(classes[img])+'\n')

#######################################################
# building dirs tree on the dst_ft folder
print('Building ft tree')
dst_ft = os.path.join(dst_root, 'ft_256')
dst_ft = Path(dst_ft)
dst_ft.mkdir(parents=True, exist_ok=True)
with open(os.path.join(src_root, 'categories_places365.txt')) as f:
    for cat in f.readlines():
        cat_folder = (cat.strip().split()[0])[1:]
        dst_folder = os.path.join(dst_ft, cat_folder)
        os.makedirs(dst_folder, exist_ok=True)

print('Copying ft files')
indexes = []
for label in range(365):
    indexes.append(np.argwhere(classes==label).squeeze()[2000:2150])
    for img in indexes[label]:
        src = os.path.join(src_train, paths[img])
        dst = os.path.join(dst_ft, paths[img])
        copyfile(src, dst)

# writing the classes file
print('Writing ft file')
dst_ft_file = os.path.join(dst_root, 'places365_ft.txt')
with open(dst_ft_file, 'w') as f:
    for label in range(365):
        for img in indexes[label]:
            f.write(paths[img]+' '+str(classes[img])+'\n')

#################################################################
# building dirs tree on the dst_ft folder
print('Building trainft tree')
dst_trainft = os.path.join(dst_root, 'trainft_256')
dst_trainft = Path(dst_trainft)
dst_trainft.mkdir(parents=True, exist_ok=True)

with open(os.path.join(src_root, 'categories_places365.txt')) as f:
    for cat in f.readlines():
        cat_folder = (cat.strip().split()[0])[1:]
        dst_folder = os.path.join(dst_trainft, cat_folder)
        os.makedirs(dst_folder, exist_ok=True)

print('Copying trainft files')
indexes = []
for label in range(365):
    indexes.append(np.argwhere(classes==label).squeeze()[:2150])
    for img in indexes[label]:
        src = os.path.join(src_train, paths[img])
        dst = os.path.join(dst_trainft, paths[img])
        copyfile(src, dst)

# writing the classes file
print('Writing ft file')
dst_trainft_file = os.path.join(dst_root, 'places365_trainft.txt')
with open(dst_trainft_file, 'w') as f:
    for label in range(365):
        for img in indexes[label]:
            f.write(paths[img]+' '+str(classes[img])+'\n')

#####################################################
# building dirs tree on the dst_ft folder
print('Building test tree')
dst_test = os.path.join(dst_root, 'test_256')
dst_test = Path(dst_test)
dst_test.mkdir(parents=True, exist_ok=True)
with open(os.path.join(src_root, 'categories_places365.txt')) as f:
    for cat in f.readlines():
        cat_folder = (cat.strip().split()[0])[1:]
        dst_folder = os.path.join(dst_test, cat_folder)
        os.makedirs(dst_folder, exist_ok=True)

print('Copying test files')
indexes = []
for label in range(365):
    indexes.append(np.argwhere(classes==label).squeeze()[2150:2650])
    for img in indexes[label]:
        src = os.path.join(src_train, paths[img])
        dst = os.path.join(dst_test, paths[img])
        copyfile(src, dst)

# writing the classes file
print('Writing test file')
dst_test_file = os.path.join(dst_root, 'places365_test.txt')
with open(dst_test_file, 'w') as f:
    for label in range(365):
        for img in indexes[label]:
            f.write(paths[img]+' '+str(classes[img])+'\n')

#####################################################
# building dirs tree on the dst_ft folder
print('Building val tree')
dst_val = os.path.join(dst_root, 'val_256')
dst_val = Path(dst_val)
dst_val.mkdir(parents=True, exist_ok=True)
with open(os.path.join(src_root, 'categories_places365.txt')) as f:
    for cat in f.readlines():
        cat_folder = (cat.strip().split()[0])[1:]
        dst_folder = os.path.join(dst_val, cat_folder)
        os.makedirs(dst_folder, exist_ok=True)

print('Copying val files')
indexes = []
for label in range(365):
    indexes.append(np.argwhere(classes==label).squeeze()[2650:2850])
    for img in indexes[label]:
        src = os.path.join(src_train, paths[img])
        dst = os.path.join(dst_val, paths[img])
        copyfile(src, dst)

# writing the classes file
print('Writing val file')
dst_val_file = os.path.join(dst_root, 'places365_val.txt')
with open(dst_val_file, 'w') as f:
    for label in range(365):
        for img in indexes[label]:
            f.write(paths[img]+' '+str(classes[img])+'\n')

#####################################################
# copy validation and test files
# val_folder = 'val_256'
# copy_tree(os.path.join(src_root, val_folder), os.path.join(dst_root, val_folder))

# val_file = 'places365_val.txt'
# copyfile(os.path.join(src_root, val_file), os.path.join(dst_root, val_file))

