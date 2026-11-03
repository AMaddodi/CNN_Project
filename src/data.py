import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# CIFAR-10 channel statistics
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_transforms(augment:bool):
    """ Transform function based on augmentation """
    norm = [T.ToTensor(), T.Normalize(MEAN,STD)]
    if augment:
        aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
        return T.Compose(aug + norm)
    return T.Compose(norm)

def get_loader(root='./data', batch_size=128, val_size=5000, augment=True, num_worker=2, seed=42):
    """
    Return train/val/test loaders.
    val is carved out of the training set, so the test set is only used once.
    """
    train_full = torchvision.datasets.CIFAR10(root,
                                              train=True,
                                              download=True,
                                              transform=get_transforms(augment))
    # same images but without augmentation
    val_full = torchvision.datasets.CIFAR10(root,
                                            train=True,
                                            download=True,
                                            transform=get_transforms(False))
    test_set = torchvision.datasets.CIFAR10(root,
                                            train=False,
                                            download=True,
                                            transform=get_transforms(False))

    g = torch.Generator().manual_seed(seed)
    prem = torch.randperm(len(train_full), generator=g).tolist()
    val_idx, train_idx = prem[:val_size], prem[val_size:]

    train_loader = DataLoader(Subset(train_full, train_idx),
                              batch_size = batch_size,
                              shuffle=True,
                              num_workers=num_worker,
                              pin_memory=True)
    val_loader = DataLoader(Subset(val_full,val_idx),
                            batch_size= 256,
                            shuffle=False,
                            num_workers=num_worker,
                            pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=256,
                             shuffle=False,
                             num_workers=num_worker,
                             pin_memory=True)
    return train_loader, val_loader, test_loader

def denormalize(img: torch.Tensor) -> torch.Tensor:
    """(3,H,W), normalized tensor -> (H,W,3) image in [0,1] for plotting """
    m = torch.tensor(MEAN).view(3,1,1)
    s = torch.tensor(STD).view(3,1,1)
    return (img.detach().cpu()*s+m).clamp(0,1).permute(1,2,0)



