import os
import sys
import copy
import argparse
from pathlib import Path

# Add project root to PYTHONPATH automatically
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn as nn

from src.utils import load_config, seed_everything, get_device, get_logger, print_system_info
from src.dataset import build_pairs, split_pairs, sanity_check_pairs, make_loaders
from src.models import create_model
from src.evaluate import predict_loader, evaluate_robustness, plot_reliability
from src.metrics import find_best_threshold
from src.calibration import fit_temperature

def train_one_model(model_name, train_loader, val_loader, config, device, logger):
    model = create_model(model_name).to(device)
    opt = torch.optim.Adamax(model.parameters(), lr=float(config["training"]["lr"]), 
                             weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=15)
    ce = nn.CrossEntropyLoss()
    
    best_state = copy.deepcopy(model.state_dict())
    best_auc, best_epoch, patience = -1.0, 0, 0
    history = []
    
    epochs = config["training"]["epochs"]
    max_patience = config["training"]["patience"]

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
            
            loss_sum += loss.item() * x.size(0)
            count += x.size(0)

        val_metrics, val_y, val_prob, _ = predict_loader(model, val_loader, device, threshold=0.5)
        best_thr, best_bal = find_best_threshold(val_y, val_prob)
        train_loss = loss_sum / max(count, 1)
        
        hist_entry = {
            "epoch": epoch, "train_loss": train_loss, 
            "val_best_threshold": best_thr, "val_best_balanced_accuracy": best_bal,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        history.append(hist_entry)
        
        auc = val_metrics["roc_auc"]
        logger.info(f"Epoch {epoch:03d} | loss={train_loss:.4f} | val_auc={auc:.4f} | val_acc={val_metrics['accuracy']:.4f} | val_bal_best={best_bal:.4f} | thr={best_thr:.3f} | lr={opt.param_groups[0]['lr']:.2e}")
        
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc, best_epoch, patience = auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            
        if patience >= max_patience:
            logger.info(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}, best AUC={best_auc:.4f}")
            break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

def run_single_experiment(experiment_key, models, config, device, logger):
    logger.info("=" * 80)
    logger.info(f"Starting Experiment: {experiment_key}")
    logger.info("=" * 80)

    exp_dict = config["experiments"][experiment_key]
    cover_dir = Path(exp_dict["cover"])
    stego_dir = Path(exp_dict["stego"])

    pairs = build_pairs(cover_dir, stego_dir)
    sanity_check_pairs(pairs, n=10)

    train_pairs, val_pairs, test_pairs = split_pairs(pairs, config)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = make_loaders(
        train_pairs, val_pairs, test_pairs, config, device
    )

    exp_dir = Path(config["training"]["out_dir"]) / experiment_key
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    master_results = []

    for model_name in models:
        logger.info("-" * 80)
        logger.info(f"Training: {model_name} | {experiment_key}")
        logger.info("-" * 80)

        seed_everything(config["training"]["seed"])
        
        model, hist = train_one_model(model_name, train_loader, val_loader, config, device, logger)
        hist_path = exp_dir / f"{model_name}_{experiment_key}_history.csv"
        hist.to_csv(hist_path, index=False)

        T = fit_temperature(model, val_loader, device)
        val_metrics, val_y, val_prob, _ = predict_loader(model, val_loader, device, temperature=T, threshold=0.5)
        best_thr, best_bal = find_best_threshold(val_y, val_prob)
        
        logger.info(f"Temperature: {T:.4f}")
        logger.info(f"Best validation threshold: {best_thr:.4f} | Best val balanced accuracy: {best_bal:.4f}")

        test_metrics, test_y, test_prob, test_paths = predict_loader(
            model, test_loader, device, temperature=T, threshold=best_thr
        )
        logger.info(f"Test metrics: {test_metrics}")

        # Save Checkpoint
        ckpt_path = exp_dir / f"{model_name}_{experiment_key}.pt"
        torch.save({
            "model_name": model_name, "experiment_key": experiment_key, 
            "state_dict": model.state_dict(), "temperature": T, 
            "threshold": best_thr, "config": config
        }, ckpt_path)

        # Plot Reliability
        plot_reliability(test_y, test_prob, f"{model_name} / {experiment_key}",
                         exp_dir / f"{model_name}_{experiment_key}_reliability.png")

        # Robustness
        logger.info("Evaluating robustness...")
        rob_df = evaluate_robustness(model, test_ds, config, device, T, best_thr, logger)
        rob_df.to_csv(exp_dir / f"{model_name}_{experiment_key}_robustness.csv", index=False)
        
        test_metrics_df = pd.DataFrame([test_metrics])
        test_metrics_df.to_csv(exp_dir / f"{model_name}_{experiment_key}_test_metrics.csv", index=False)
        
        master_results.append({
            "model": model_name,
            "experiment": experiment_key,
            **test_metrics
        })

    master_df = pd.DataFrame(master_results)
    master_df.to_csv(exp_dir / f"master_summary_results.csv", index=False)
    logger.info("Experiment finished successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.debug:
        config["run"]["debug"] = True
    
    experiments_to_run = [args.experiment] if args.experiment else config["run"]["experiment_keys"]
    models_to_run = [args.model] if args.model else config["run"]["models"]

    Path(config["training"]["out_dir"]).mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_file=Path(config["training"]["out_dir"]) / "training.log")
    
    device = get_device()
    print_system_info()
    logger.info(f"Running experiments: {experiments_to_run}")
    logger.info(f"Running models: {models_to_run}")
    logger.info(f"Debug mode: {config['run']['debug']}")

    for exp in experiments_to_run:
        run_single_experiment(exp, models_to_run, config, device, logger)

if __name__ == "__main__":
    main()
