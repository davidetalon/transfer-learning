from pathlib import Path
import shutil

def gather_data():

     # reading images ids with paths
    with open('data/raw/CUB_200_2011/images.txt') as f:
        data = f.read()
        data = data.split()

    # reading ids and is_training flag
    with open('data/raw/CUB_200_2011/train_test_split.txt') as f:
        is_training = f.read()
        is_training = is_training.split()

    # gather dataset data
    dataset = []
    for image in range(len(data)//2):
        data_dict = {'id':int(data[image*2]), 'path':data[image*2 + 1], 'class': data[image*2 + 1][:3],'is_training':int(is_training[image*2+1])}
        dataset.append(data_dict)
    
    with open('data/raw/CUB_200_2011/classes.txt') as f:
        classes = f.read()
        classes = classes.split()
        cl=[]
        for idx in range(len(classes)//2):
            cl.append(classes[idx * 2 + 1])


    return dataset, cl

def mk_class_dirs(classes, train_folder, test_folder):

    # create folders
    for cl in classes:

        # train
        train_class_folder = Path(cl)
        train_class_folder = train_folder/train_class_folder
        train_class_folder.mkdir(parents=True, exist_ok=True)

        # test
        test_class_folder = Path(cl)
        test_class_folder = test_folder/test_class_folder
        test_class_folder.mkdir(parents=True, exist_ok=True)



def copyfiles(dataset, raw_data_folder, train_folder, test_folder):
    for img in dataset:
            src = raw_data_folder / img['path']
            if img['is_training']:
                dest = train_folder /  img['path']
            else:
                dest = test_folder /  img['path']
            shutil.copy2(src, dest)



def train_test_split(raw_data_folder, processed_data_folder):

    train_folder = 'train'
    test_folder = 'test'

    raw_data_folder = Path(raw_data_folder)

    # creating folders
    processed_data_folder = Path(processed_data_folder)
    processed_data_folder.mkdir(parents=True, exist_ok=True)

    # train
    train_folder = Path(train_folder)
    train_folder = processed_data_folder / train_folder
    train_folder.mkdir(parents=True, exist_ok=True)

    # test
    test_folder = Path(test_folder)
    test_folder = processed_data_folder / test_folder
    test_folder.mkdir(parents=True, exist_ok=True)

    dataset, classes = gather_data()

    mk_class_dirs(classes, train_folder, test_folder)

    copyfiles(dataset, raw_data_folder, train_folder, test_folder)


if __name__=='__main__':

    train_test_split('data/raw/CUB_200_2011/images', 'data/processed')
    


    

    


        

    
    