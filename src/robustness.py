import io
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import torch
from torch.utils.data import Dataset, DataLoader

class RobustnessDataset(Dataset):
    def __init__(self, base_dataset, transform_name, image_size=256):
        self.base_dataset = base_dataset
        self.transform_name = transform_name
        self.image_size = image_size

    def __len__(self):
        return len(self.base_dataset)

    def apply_transform(self, img):
        t = self.transform_name
        if t == "clean":
            return img
        if t.startswith("jpeg_q"):
            q = int(t.replace("jpeg_q", ""))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return Image.open(buf).convert("L")
        if t == "resize_down_up":
            small = img.resize((self.image_size // 2, self.image_size // 2), Image.BICUBIC)
            return small.resize((self.image_size, self.image_size), Image.BICUBIC)
        if t == "gaussian_blur":
            return img.filter(ImageFilter.GaussianBlur(radius=1.0))
        if t == "gaussian_noise":
            arr = np.array(img, dtype=np.float32)
            noise = np.random.default_rng(123).normal(0, 5, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        raise ValueError(t)

    def __getitem__(self, idx):
        path, label = self.base_dataset.samples[idx]
        img = Image.open(path).convert("L")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            
        img = self.apply_transform(img)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr[None, :, :]), torch.tensor(label, dtype=torch.long), str(path)
