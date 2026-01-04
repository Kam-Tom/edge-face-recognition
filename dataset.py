import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

STATS = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

def get_transforms(is_train=True):
    t_list = [transforms.Resize((112, 112))]
    if is_train:
        t_list.append(transforms.RandomHorizontalFlip(p=0.5))
        t_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
    t_list.extend([transforms.ToTensor(), transforms.Normalize(*STATS)])
    return transforms.Compose(t_list)

def get_dataloaders(root_dir, batch_size=32, val_split=0.2, num_workers=2):
    print(f"🔄 Loading data from: {root_dir}")
    
    full_dataset = datasets.ImageFolder(root_dir)
    targets = full_dataset.targets

    # Stratified split - keeps class proportions balanced
    try:
        train_idx, val_idx = train_test_split(
            np.arange(len(targets)), 
            test_size=val_split, 
            shuffle=True, 
            stratify=targets
        )
    except ValueError:
        print("⚠️ Some class has only 1 image - remove folders with <2 images.")
        return None, None, 0

    train_subset = Subset(datasets.ImageFolder(root_dir, transform=get_transforms(True)), train_idx)
    val_subset = Subset(datasets.ImageFolder(root_dir, transform=get_transforms(False)), val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    print(f"✅ Split: {len(train_idx)} train, {len(val_idx)} val | Classes: {len(full_dataset.classes)}")
    
    return train_loader, val_loader, len(full_dataset.classes)