import os
import numpy as np
import torch
from torch.utils.data import Dataset


def getDataloaders(cfg):
    ### Data Loading ###
    print('Load dataset ...')
    cfg.dataset.train_labels = ['train_labels.npy']*len(cfg.dataset.train_tasks)
    cfg.dataset.test_labels =['test_labels.npy']*len(cfg.dataset.test_tasks)

    TRAIN_DATASET = ModelNetDataLoader(root=cfg.dataset.data_root, 
                                   tasks=cfg.dataset.train_tasks,
                                   labels=cfg.dataset.train_labels,
                                   partition='train',
                                   npoint=cfg.dataset.num_point,      
                                   normal_channel=cfg.dataset.normal)
    TEST_DATASET = ModelNetDataLoader(root=cfg.dataset.data_root, 
                                   tasks=cfg.dataset.test_tasks,
                                   labels=cfg.dataset.test_labels,
                                   partition='test',
                                   npoint=cfg.dataset.num_point,      
                                   normal_channel=cfg.dataset.normal)
    train_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    test_loader = None

    print("train loader:", len(train_loader), \
          "valid loader:", len(valid_loader), \
        )

    return train_loader, valid_loader, test_loader

class ModelNetDataLoader(Dataset):
    def __init__(self, root, tasks, labels, partition='train', npoint=2048, uniform=False, normal_channel=False, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.normal_channel = normal_channel
        self.data, self.label = load_data(root, tasks, labels)
        self.partition = partition
        print('The number of ' + partition + ' data: ' + str(self.data.shape[0]))

    def __len__(self):
        return self.data.shape[0]
    
    def _get_item(self, index):
        pointcloud = self.data[index][:self.npoints]
        label = self.label[index]
        if self.partition == 'train':
            np.random.shuffle(pointcloud)

        return pointcloud, label

    def __getitem__(self, index):
        return self._get_item(index)

def load_data(root, tasks, labels):
    all_data = []
    all_label = []
    for i in range(len(tasks)):
        data = np.load(os.path.join(root, tasks[i]))
        label = np.load(os.path.join(root, labels[i]))
        
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label