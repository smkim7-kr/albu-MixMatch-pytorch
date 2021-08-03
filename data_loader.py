import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_norm(args):
    if args.datasets == 'CIFAR10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif args.datasets == 'CIFAR100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    elif args.datasets == 'SVHN':
        mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
    elif args.datasets == 'STL10':
        mean, std = (0.4409, 0.4279, 0.3868), (0.2683, 0.2611, 0.2687)
    else:
        raise Exception('unknown datasets: {}'.format(args.datasets))
    return mean, std

#transformations
def transforms(_dataset, mean, std):
    if 'CIFAR' in _dataset:
        transform_train = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Resize(40, 40),
                A.RandomCrop(32, 32), 
                A.Normalize(mean, std),
                ToTensorV2()
            ])
        transform_val = A.Compose([
                A.Normalize(mean, std),
                ToTensorV2()
            ])
    elif _dataset == 'SVHN':
        transform_train = A.Compose([
                A.Resize(40, 40),
                A.RandomCrop(32, 32), 
                A.Normalize(mean, std),
                ToTensorV2()
            ])

        transform_val = A.Compose([
                A.Normalize(mean, std),
                ToTensorV2()
            ])
    elif _dataset == 'STL10':
        transform_train = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Resize(112, 112),
                A.RandomCrop(96, 96), 
                A.Normalize(mean, std),
                ToTensorV2()
            ])
        transform_val = A.Compose([
                A.Normalize(mean, std),
                ToTensorV2()
            ])
    return transform_train, transform_val

class stochasticAug:
    def __init__(self, K, transform):
        self.transform = transform
        self.K = K

    def __call__(self, image):
        out = [self.transform(image=image)["image"] for _ in range(self.K)]
        return out

def train_val_split(labels, n_labeled, n_class, n_val):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(n_class):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled//n_class])
        train_unlabeled_idxs.extend(idxs[n_labeled//n_class:-n_val])
        val_idxs.extend(idxs[-n_val:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

class SSL_Dataset(data.Dataset):
    def __init__(self, data, targets, idx=None, transform=None, islabel=True):
        self.data = data
        self.targets = targets
        if idx is not None:
            self.data = self.data[idx]
            self.targets = np.array(self.targets)[idx]
        self.transform = transform
        self.islabel = islabel
  
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            if self.islabel:
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(image=img)
                target = -1
    
        return img, target
    def __len__(self):
        return len(self.data)

def load_dataloader(args):
    if args.datasets == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        train_l_idx, train_u_idx, val_idx = train_val_split(dataset.targets, 250, 10, 500)
        mean, std = get_norm(args)
        transform_train, transform_val = transforms(args.datasets, mean, std)

        train_labeled_dataset = SSL_Dataset(dataset.data, dataset.targets, train_l_idx, transform=transform_train, islabel=True)
        train_unlabeled_dataset = SSL_Dataset(dataset.data, dataset.targets, train_u_idx, transform=stochasticAug(args.K,transform_train), islabel=False)
        val_dataset = SSL_Dataset(dataset.data, dataset.targets, val_idx, transform=transform_val, islabel=True)
        test_dataset = SSL_Dataset(test_dataset.data, test_dataset.targets, None, transform=transform_val, islabel=True)
    elif args.datasets == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        train_l_idx, train_u_idx, val_idx = train_val_split(dataset.targets, 250, 10, 500)
        mean, std = get_norm(args)
        transform_train, transform_val = transforms(args.datasets, mean, std)

        train_labeled_dataset = SSL_Dataset(dataset.data, dataset.targets, train_l_idx, transform=transform_train, islabel=True)
        train_unlabeled_dataset = SSL_Dataset(dataset.data, dataset.targets, train_u_idx, transform=stochasticAug(args.K,transform_train), islabel=False)
        val_dataset = SSL_Dataset(dataset.data, dataset.targets, val_idx, transform=transform_val, islabel=True)
        test_dataset = SSL_Dataset(test_dataset.data, test_dataset.targets, None, transform=transform_val, islabel=True)
    elif args.datasets == 'SVHN':
        dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True)
        test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True)
        train_l_idx, train_u_idx, val_idx = train_val_split(dataset.labels, 250, 10, 500)
        mean, std = get_norm(args)
        transform_train, transform_val = transforms(args.datasets, mean, std)

        train_labeled_dataset = SSL_Dataset(dataset.data, dataset.targets, train_l_idx, transform=transform_train, islabel=True)
        train_unlabeled_dataset = SSL_Dataset(dataset.data, dataset.targets, train_u_idx, transform=stochasticAug(args.K,transform_train), islabel=False)
        val_dataset = SSL_Dataset(dataset.data, dataset.targets, val_idx, transform=transform_val, islabel=True)
        test_dataset = SSL_Dataset(test_dataset.data, test_dataset.targets, None, transform=transform_val, islabel=True)
    elif args.datasets == 'STL10':
        dataset = torchvision.datasets.STL10(root='./data', split='train', download=True)
        test_dataset = torchvision.datasets.STL10(root='./data', split='test', download=True)
        train_l_idx, train_u_idx, val_idx = train_val_split(dataset.targets, 250, 10, 500)
        mean, std = get_norm(args)
        transform_train, transform_val = transforms(args.datasets, mean, std)

        train_labeled_dataset = SSL_Dataset(dataset.data, dataset.targets, train_l_idx, transform=transform_train, islabel=True)
        train_unlabeled_dataset = SSL_Dataset(dataset.data, dataset.targets, train_u_idx, transform=stochasticAug(args.K,transform_train), islabel=False)
        val_dataset = SSL_Dataset(dataset.data, dataset.targets, val_idx, transform=transform_val, islabel=True)
        test_dataset = SSL_Dataset(test_dataset.data, test_dataset.targets, None, transform=transform_val, islabel=True)
    
    #Dataloader
    l_train_loader = torch.utils.data.DataLoader(dataset=train_labeled_dataset, 
                                                batch_size=args.batch_size,
                                                pin_memory=True,
                                                num_workers=args.num_workers,
                                                drop_last=True)
    u_train_loader = torch.utils.data.DataLoader(dataset=train_unlabeled_dataset, 
                                                batch_size=args.batch_size,
                                                pin_memory=True,
                                                num_workers=args.num_workers,
                                                drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size,
                                            pin_memory=True,
                                            num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=1,
                                            pin_memory=True,
                                            num_workers=args.num_workers)
                                            
    return l_train_loader, u_train_loader, val_loader, test_loader