import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics import compute_metrics
from src.robustness import RobustnessDataset

@torch.no_grad()
def predict_loader(model, loader, device, temperature=1.0, threshold=0.5):
    model.eval()
    y_all, prob_all, path_all = [], [], []
    loss_sum, count = 0.0, 0
    ce = nn.CrossEntropyLoss()
    
    for x, y, paths in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        probs = torch.softmax(logits / temperature, dim=1)[:, 1]
        
        y_all.extend(y.detach().cpu().numpy().tolist())
        prob_all.extend(probs.detach().cpu().numpy().tolist())
        path_all.extend(list(paths))
        
        loss_sum += loss.item() * x.size(0)
        count += x.size(0)
        
    metrics = compute_metrics(y_all, prob_all, threshold=threshold)
    metrics["loss"] = loss_sum / max(count, 1)
    return metrics, np.array(y_all), np.array(prob_all), path_all

def evaluate_robustness(model, test_ds, config, device, temperature, threshold, logger):
    transforms = ["clean", "jpeg_q95", "jpeg_q85", "jpeg_q75", "jpeg_q50", 
                  "resize_down_up", "gaussian_blur", "gaussian_noise"]
    rows = []
    
    for t in transforms:
        ds = RobustnessDataset(test_ds, t, config["dataset"]["image_size"])
        loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=False,
                            num_workers=config["training"]["num_workers"], pin_memory=(device.type == "cuda"))
        m, _, _, _ = predict_loader(model, loader, device, temperature=temperature, threshold=threshold)
        rows.append({"transformation": t, **m})
        logger.info(f"{t:<15} acc: {m['accuracy']:.4f} | bal: {m['balanced_accuracy']:.4f} | auc: {m['roc_auc']:.4f}")
        
    return pd.DataFrame(rows)

def plot_reliability(y_true, y_prob, title, save_path):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    conf = np.maximum(y_prob, 1 - y_prob)
    pred = (y_prob >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    
    xs, ys = [], []
    bins = np.linspace(0, 1, 11)
    for i in range(10):
        mask = (conf > bins[i]) & (conf <= bins[i+1])
        if np.any(mask):
            xs.append(np.mean(conf[mask]))
            ys.append(np.mean(correct[mask]))
            
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.scatter(xs, ys, color="blue", alpha=0.7)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
