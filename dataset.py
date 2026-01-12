import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

STATS = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

def get_transforms(is_train=True):
    t_list = [transforms.Resize((112, 112))]
    if is_train:
        t_list.append(transforms.RandomHorizontalFlip(p=0.5))
        t_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
    t_list.extend([transforms.ToTensor(), transforms.Normalize(*STATS)])
    return transforms.Compose(t_list)


class TransformSubset(Dataset):
    """Subset with transform applied on-the-fly."""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)  # Convert to list for pickling
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(root_dir, batch_size=32, val_split=0.2, num_workers=None):
    print(f"🔄 Loading data from: {root_dir}")
    
    if num_workers is None:
        available_cpu = os.cpu_count() or 4
        # Use 75% of CPUs, max 24, min 4
        num_workers = min(max(available_cpu * 3 // 4, 4), 24)
    
    # Load dataset ONCE without transforms
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

    # Apply different transforms to train/val subsets
    train_subset = TransformSubset(full_dataset, train_idx, get_transforms(True))
    val_subset = TransformSubset(full_dataset, val_idx, get_transforms(False))

    # Optimize DataLoader for faster loading
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    
    print(f"✅ Split: {len(train_idx)} train, {len(val_idx)} val | Classes: {len(full_dataset.classes)} | Workers: {num_workers}")
    print(f"⚠️  Note: Train/val share identities. Use separate test set with different people for final eval.")
    
    return train_loader, val_loader, len(full_dataset.classes)