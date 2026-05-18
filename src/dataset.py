import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

IMAGE_EXTS = [".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def is_image_file(p: Path):
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def list_image_files(folder: Path):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted([p for p in folder.iterdir() if is_image_file(p)])

def build_pairs(cover_dir: Path, stego_dir: Path):
    cover_files = list_image_files(cover_dir)
    stego_files = list_image_files(stego_dir)
    stego_by_stem = {s.stem: s for s in stego_files}

    pairs, missing = [], 0
    for c in cover_files:
        s = stego_by_stem.get(c.stem)
        if s is None:
            missing += 1
        else:
            pairs.append((c, s))

    print("Cover images:", len(cover_files))
    print("Stego images:", len(stego_files))
    print("Matched pairs:", len(pairs))
    print("Missing cover matches:", missing)

    if not pairs:
        raise RuntimeError("No matched pairs found. Check filenames and folder structure.")
    return pairs

def split_pairs(pairs, config):
    rng = random.Random(config["training"]["seed"])
    pairs = pairs.copy()
    rng.shuffle(pairs)

    if config["run"]["debug"]:
        n = min(config["dataset"]["debug_pairs"], len(pairs))
        pairs = pairs[:n]
        n_train = int(0.70 * n)
        n_val = int(0.10 * n)
        return pairs[:n_train], pairs[n_train:n_train+n_val], pairs[n_train+n_val:]

    total = config["dataset"]["train_pairs"] + config["dataset"]["val_pairs"] + config["dataset"]["test_pairs"]
    if len(pairs) < total:
        raise RuntimeError(f"Need {total} pairs but found {len(pairs)}.")
        
    train_pairs = config["dataset"]["train_pairs"]
    val_pairs = config["dataset"]["val_pairs"]
    test_pairs = config["dataset"]["test_pairs"]
    
    return (
        pairs[:train_pairs],
        pairs[train_pairs:train_pairs+val_pairs],
        pairs[train_pairs+val_pairs:train_pairs+val_pairs+test_pairs],
    )

def sanity_check_pairs(pairs, n=10):
    rng = random.Random(123)
    sample = rng.sample(pairs, min(n, len(pairs)))
    rows = []
    
    all_zero_changed = True
    for c, s in sample:
        ca = np.array(Image.open(c).convert("L"), dtype=np.uint8)
        sa = np.array(Image.open(s).convert("L"), dtype=np.uint8)
        diff = ca.astype(np.int16) - sa.astype(np.int16)
        
        changed_pct = float(np.mean(ca != sa) * 100.0)
        if changed_pct > 0:
            all_zero_changed = False
            
        rows.append({
            "cover": c.name,
            "stego": s.name,
            "changed_pixels_pct": changed_pct,
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "max_abs_diff": int(np.max(np.abs(diff))),
        })
    df = pd.DataFrame(rows)
    print("Sanity Check Pairs:\n", df)
    
    if all_zero_changed and len(sample) > 0:
        raise ValueError("FATAL: changed_pixels_pct is 0 for all checked samples! Stego images are identical to cover.")
    return df

class PairedStegoDataset(Dataset):
    def __init__(self, pairs, image_size=256, train=False, augment=True):
        self.samples = []
        for c, s in pairs:
            self.samples.append((c, 0))
            self.samples.append((s, 1))
        self.image_size = image_size
        self.train = train
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr[None, :, :])

        if self.train and self.augment:
            k = random.randint(0, 3)
            x = torch.rot90(x, k, dims=[1, 2])
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])

        return x, torch.tensor(label, dtype=torch.long), str(path)

def make_loaders(train_pairs, val_pairs, test_pairs, config, device):
    image_size = config["dataset"]["image_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    
    train_ds = PairedStegoDataset(train_pairs, image_size, train=True, augment=True)
    val_ds = PairedStegoDataset(val_pairs, image_size, train=False, augment=False)
    test_ds = PairedStegoDataset(test_pairs, image_size, train=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == "cuda"))

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))
    print("Test samples:", len(test_ds))
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
