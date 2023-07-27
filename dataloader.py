from torchvision import transforms,datasets
import torch
import numpy as np
from torch.utils.data import DataLoader
import random, copy
import os
from utils import *

data_root = "/home/pratyus2/scratch/projects/layer_learning/data"

def seed_everything(seed: int):
    # print("setting seed", seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.use_deterministic_algorithms()


cifar100_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
cifar100_label_to_idx = {label:i for i, label in enumerate(cifar100_labels)}

orthogonal_classes = ['apple', 'baby', 'bed', 'bottle', 'keyboard', 'lamp', 'sunflower', 'skyscraper', 'cloud', 'road']


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split):
        root = data_path
        self.split = split
        self.data = torch.load(f"{root}/{split}_x.pt")
        self.targets = torch.load(f"{root}/{split}_y.pt").long()
        self.n_classes = torch.unique(self.targets).shape[0]
        self.transform = None

    def __getitem__(self, index):
        x_data_index = self.data[index]
        if self.transform:
            x_data_index = self.transform(x_data_index)
        return (x_data_index, self.targets[index], index)

    def __len__(self):
        return self.data.shape[0]


def get_split_ids(dataset_size, ratio):
    indices = list(range(dataset_size))
    random.Random(0).shuffle(indices)
    split = int(dataset_size*ratio)
    pre_indices, ft_indices = indices[split:], indices[:split]
    pre_indices.sort()
    ft_indices.sort()
    # ipdb.set_trace()
    return pre_indices, ft_indices

def corrupt_labels(dset, n_classes, corrupt_prob, seed = 0, label_noise = True):
    labels = np.array(dset.targets)
    #Intialise a random number generator
    rng = np.random.default_rng(seed)
    # mask = rng.random(len(labels)) <= corrupt_prob
    num_examples = int(corrupt_prob*len(labels))
    idx = rng.choice(np.arange(len(labels)), num_examples, replace = False)
    mask = np.zeros(len(labels)).astype('int64')
    mask[idx] = 1
    if label_noise:
        #Random label should not coincide with true label
        if n_classes != 2: rnd_labels = rng.choice(n_classes - 2, num_examples) + 1 #we will do [(true + rand) % num_classes]
        else: rnd_labels = 1
        labels[idx] = (labels[idx] + rnd_labels) % n_classes
    else:
        rnd_labels = rng.choice(n_classes, num_examples)
        labels[idx] = rnd_labels
    labels = [int(x) for x in labels]
    dset.targets = labels
    try:
        dset.labels = labels
    except: 
        a = 1
    return dset, mask

call_dataset = {"mnist":datasets.MNIST,  
                "cifar10":datasets.CIFAR10, 
                "svhn":datasets.SVHN
                }

def return_basic_dset(dataset, split, log_factor=2, seed_superclass = 1, aug = True):
    train = True if split == "tr" else False
    dims = 1 if dataset in ["mnist"] else 3
    normalize = (0.5,)*dims

    if dataset in ["mnist", "svhn"]:
        tvs_train = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize(normalize, normalize),
                            ])
    else:
        tvs_train =   transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(normalize, normalize),
                            ])

    
    tvs_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(normalize, normalize),
                ])


    if split == "tr" and aug:
        # tvs = tvs_train 
        #use AutoAugment
        print ("Using AutoAugment")
        # if dataset == "cifar10":
        policy = torchvision.transforms.AutoAugmentPolicy.CIFAR10
        tvs =transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.AutoAugment(policy=policy),
                                    transforms.ToTensor(),
                                    transforms.Normalize(normalize, normalize),
                                    ])
            # import ipdb; ipdb.set_trace()
            # print(policy)
        # elif dataset == "svhn":
            # policy = torchvision.transforms.AutoAugmentPolicy.SVHN
        # tvs = transforms.Compose([tvs, torchvision.transforms.AutoAugment(policy=policy)])


    else:
        tvs = tvs_test

    if dataset == 'svhn':
        dset = dataset_with_indices(call_dataset[dataset])(f'{data_root}', download=False, split = 'train' if train else 'test', transform = tvs)
        dset.targets = dset.labels
    else:
        dset = dataset_with_indices(call_dataset[dataset])(f'{data_root}', download=False, train=train, transform=tvs)

    try:
        n_classes = torch.tensor(dset.targets).max().item() + 1
    except:
        n_classes = dset.targets.max().item() + 1
    
    return n_classes, dset

def get_dset(split, dataset, noise_ratio, indices, minority_ratio = 0, seed = 0, log_factor = 2, seed_superclass=1, split_ratio = 0.5, aug = True):
    n_classes, dset = return_basic_dset(dataset, split, log_factor, seed_superclass, aug)

    #get the correct slice
    if indices is not None:
        pre_indices, ft_indices =  get_split_ids(dset.data.shape[0], ratio = split_ratio)
        # print("Num indices less than 23446 = ", (torch.tensor(pre_indices) < 23446).sum().item())
        indices = pre_indices if indices == "pre" else ft_indices
        dset.data = dset.data[indices]
        try: dset.targets = dset.targets[indices]
        except: dset.targets = torch.tensor(dset.targets)[indices] #for cifar10
                

    mask, mask2 = None, None
    if noise_ratio > 0: 
        dset, mask = corrupt_labels(dset, n_classes, noise_ratio, seed)
    if minority_ratio > 0: 
        assert("Not Implemented")
    return dset, mask, mask2


def adjust_for_cscore(dset, mask, cscores, seed = 0, dataset = "cifar10"):
    if dataset == "cifar10":
        cscores_file = "data/cifar10-cscores-orig-order.npz"
        #load cscores
        cscores_file = np.load(cscores_file)

        #npz to np array
        cscores = cscores_file["scores"]

        labels = cscores_file["labels"]
        orig_labels = dset.targets
        assert(np.array_equal(labels, orig_labels))

        #if cscore is <0.5 set the mask to 1
        mask = (cscores < 0.5).astype(int)
        #no change to dataset needed for cifar10
        return dset, mask
    else:
        assert(dataset == "mnist")
        #load cscores
        cscores = "data/cscores.npy"
        #load cscores
        cscores = np.load(cscores)
        #load x, y based on tf ids
        dset.data = np.load("data/mnist_byte_images.npy")
        dset.targets = np.load("data/mnist_int_labels.npy")
        #if cscore is <0.5 set the mask to 1
        mask = (cscores < 0.5).astype(int)
        return dset, mask

def return_loaders(all_args, get_frac = True, shuffle = True, split_ratio = 0.5, aug = True):
    #datasets = [mnist_b_cifar, mnist_r_cifar, mnist_cifar, mnist, cifar10, fashionmnist] #blank, random, standard
    split = "tr"
    indices1, indices2 = ("pre", "ft") if get_frac else (None, None)
    batch_size = all_args["batch_size"]
    
    d1_tr, mask_noise1, mask_rare1 = get_dset(split, all_args["dataset1"], all_args["noise_1"], indices1, all_args["minority_1"], all_args["seed"], all_args["log_factor"], all_args["seed_superclass"], split_ratio= split_ratio, aug = aug)
    
    if all_args["cscore"] > 0: 
        d1_tr, mask_noise1 = adjust_for_cscore(d1_tr, mask_noise1)
    
    preloader = DataLoader(dataset=d1_tr, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    
    if get_frac:
        d2_tr, mask_noise2, mask_rare2 = get_dset(split, all_args["dataset2"], all_args["noise_2"], indices2 , all_args["minority_2"], all_args["seed"], all_args["log_factor"], all_args["seed_superclass"], split_ratio= split_ratio, aug = aug)
        ftloader = DataLoader(dataset=d2_tr, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    else:
        d2_tr, mask_noise2, mask_rare2 = None, None, None
        ftloader = None

    

    #get test datasets
    split  = "te"
    d1, _, _ = get_dset(split, all_args["dataset1"], 0, None)
    preloader_test = DataLoader(dataset=d1, batch_size=batch_size, shuffle=False, num_workers=2)
    if get_frac:
        d2, _, _ = get_dset(split, all_args["dataset2"], 0, None)
        ftloader_test = DataLoader(dataset=d2, batch_size=batch_size, shuffle=False, num_workers=2)
    else: 
        d2 = None
        ftloader_test = None


    pre_dict = { "train_loader":preloader, 
                "test_loader":preloader_test,
                "noise_mask":mask_noise1,
                "rare_mask":mask_rare1,
                "train_dataset": d1_tr
              }

    ft_dict = { "train_loader":ftloader, 
                "test_loader":ftloader_test,
                "noise_mask":mask_noise2,
                "rare_mask":mask_rare2,
                "train_dataset": d2_tr
              }

    return pre_dict, ft_dict

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    # def __init__()
    # self.indices = torch.arange(self.targets.shape[0])

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })
