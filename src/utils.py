import os
import random
import logging
import platform
from pathlib import Path

import numpy as np
import torch
import yaml

def get_logger(name="steganalysis", log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def seed_everything(seed=2026):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def print_system_info():
    print("System:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    print("Using device:", get_device())
