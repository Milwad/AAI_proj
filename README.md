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

## Evaluation workflow
TODO – fill in once the CLI exists (step 2).

## Generation workflow
TODO – fill in once `generate` exists (step 2).

## Results (test set, baseline run)
| Metric | Value |
|---|---|
| PR-AUC (AP) | 0.721 [0.662, 0.781] |
| ROC-AUC | 0.949 [0.928, 0.966] |
| Recall / Precision @ max-F1 threshold | 0.764 / 0.834 |
