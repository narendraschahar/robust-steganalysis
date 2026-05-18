# Robust Image Steganalysis

Research implementation for covert channel detection under post-processing transformations.

## Environment Setup

We recommend using `uv` or standard `python -m venv` to create an isolated environment.

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate
# Or on Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Configuration

Ensure your dataset matches the structure or update the paths in `configs/config.yaml` explicitly. The default YAML configuration relies on paths pointing directly to `cover` and `stego` folders.

```yaml
experiments:
  wow_04:
    cover: "/path/to/BOWS2/cover"
    stego: "/path/to/BOWS2/stego/WOW/0.2bpp/stego"
```

## Running Experiments

### Debug Mode (Fast Run)
Run a quick test on 400 images using the `srm_tlu_cnn` model:

```bash
python src/train.py --experiment wow_04 --model srm_tlu_cnn --debug
```

### Full Mode
Run the complete training pipeline:

```bash
python src/train.py --experiment wow_04 --model srm_tlu_cnn
```

## Outputs

All outputs are saved to the `outputs/<experiment_key>` directory:
- `*_history.csv`: Training and validation loss/AUC metrics per epoch.
- `*.pt`: PyTorch model checkpoint.
- `*_reliability.png`: Calibration curve plotting predicted confidence against actual accuracy.
- `*_robustness.csv`: Robustness evaluation on augmentations (clean, JPEG compression, blurring, noise, resize).
- `*_test_metrics.csv`: Performance metrics on the unaugmented test set.
- `master_summary_results.csv`: Combined view of final metrics across all executed models.

If the ROC-AUC is near 0.50, review the dataset pairing script or ensure the stego directories are not empty / identical to the cover dataset.
