import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# ArcFace normalization ([-1, 1])
STATS = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

def get_transforms(is_train=True):
    t_list = []
    
    # REMOVED: transforms.Resize((112, 112)) 
    # Because data is already pre-aligned to 112x112 by prepare_data.py
    
    if is_train:
        t_list.append(transforms.RandomHorizontalFlip(p=0.5))
        # Slight color augmentation helps prevent overfitting
        t_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
    
    t_list.extend([transforms.ToTensor(), transforms.Normalize(*STATS)])
    return transforms.Compose(t_list)


class TransformSubset(Dataset):
    """Subset with transform applied on-the-fly."""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(root_dir, batch_size=32, val_split=0.2, num_workers=None):
    print(f"📂 Loading data from: {root_dir}")
    
    if num_workers is None:
        available_cpu = os.cpu_count() or 4
        num_workers = min(max(available_cpu * 3 // 4, 4), 16)
    
    # ImageFolder automatically finds classes
    full_dataset = datasets.ImageFolder(root_dir)
    targets = full_dataset.targets

    # Stratified Split (ensures every person is in both train and val if possible)
    try:
        train_idx, val_idx = train_test_split(
            np.arange(len(targets)), 
            test_size=val_split, 
            shuffle=True, 
            stratify=targets
        )
    except ValueError:
        print("⚠️ Warning: Some classes have only 1 image. Stratified split failed.")
        print("   -> Consider running clean_up script or ignore validation for those classes.")
        # Fallback to simple random split if stratification fails
        train_idx, val_idx = train_test_split(
            np.arange(len(targets)), 
            test_size=val_split, 
            shuffle=True
        )

    train_subset = TransformSubset(full_dataset, train_idx, get_transforms(True))
    val_subset = TransformSubset(full_dataset, val_idx, get_transforms(False))

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    
    print(f"✅ Split created: {len(train_idx)} train, {len(val_idx)} val images")
    print(f"👥 Total Identities: {len(full_dataset.classes)}")
    print(f"⚙️ Workers: {num_workers}, Batch Size: {batch_size}")
    
    return train_loader, val_loader, len(full_dataset.classes)