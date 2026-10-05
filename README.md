# VAE-based Credit Card Fraud Detection

IT18X57 Advanced AI – Research Assignment (2026). Theme: Generative Modelling for Anomaly Detection.

A variational autoencoder is trained on normal transactions only; fraud is flagged by a high anomaly score.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle (mlg-ulb/creditcardfraud) and place it in `data/`.

## Project structure
| File | Purpose |
|---|---|
| `src/config.py` | All hyperparameters (`Config` dataclass) |
| `src/data.py` | Loading, feature engineering, normal-only train split, scaling |
| `src/enc.py`, `src/dec.py`, `src/vae.py` | Encoder, decoder, VAE + loss |
| `src/train.py` | Training loop with KL warm-up and early stopping on val AP |
| `src/score.py` | Anomaly scoring functions |
| `src/evaluate.py` | Thresholding, metrics, bootstrap CIs, figures |

## Training workflow
Run all commands from the project root.
```bash
python src/data.py      # builds + checks the splits
python src/train.py     # trains, saves results/<run_name>/model.pt, history.json, curves.png
```

## Usage
All commands are run from the project root via one CLI. `python src/main.py <command> --help` lists every option with its default.

```bash
# training (any Config field can be overridden)
python src/main.py train --run-name baseline
python src/main.py train --run-name beta2 --beta 2 --latent-dim 16 --hidden-dims 128 64

# threshold + evaluation (reuses the run's own training config)
python src/main.py threshold --run-name baseline --threshold-method max_f1
python src/main.py evaluate  --run-name baseline --split val
python src/main.py evaluate  --run-name baseline --split test    # test set: once only

# generation: samples z ~ N(0, I), decodes, writes samples.csv, fidelity.csv, marginals.png, latent_space.png
python src/main.py generate --run-name baseline --n-samples 5000

# inference on new transactions (Class column optional)
python src/main.py predict --run-name baseline --input new_transactions.csv
```

## Results (test set, final model: beta=1, latent 32, seed 2).
| Metric | Value |
|---|---|
| PR-AUC (AP) | 0.729 [0.668, 0.794] |
| ROC-AUC | 0.963 |
| Recall / Precision @ max-F1 threshold | 0.76 / 0.891 |

## Example outputs (`inference/`)
Both examples use the final model (`experiments/beta1.0_lat32_s2`, beta=1, latent 32).

1. **`example1_generated_transactions/`**: 5,000 synthetic transactions sampled from the prior
   (`python src/main.py generate --run-name experiments/beta1.0_lat32_s2`), with per-feature fidelity
   statistics, real-vs-generated histograms and the latent space plot.
2. **`example2_fraud_detection/`**: 1,000 unseen test transactions (25 fraud) scored by
   `python src/main.py predict ...`. Each row gets an `anomaly_score`, a `flagged` decision (max-F1 threshold)
   and the three features with the largest reconstruction error (`top_features`).
