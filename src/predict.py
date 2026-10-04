import json
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from config import Config
from data import Data
from vae import load_model
from score import SCORERS
from evaluate import metrics_at_threshold

def predict(conf: Config, input_path: str, output_path: str | None = None, top_features: int = 3) -> pd.DataFrame:

    run_dir = Path(conf.results_dir) / conf.run_name
    model, checkpoint = load_model(run_dir)
    scaler = joblib.load(run_dir / "scaler.joblib")
    feature_names: list[str] = checkpoint["feature_names"]

    threshold_file = run_dir / "threshold.json"
    if not threshold_file.exists():
        raise FileNotFoundError(f"No threshold at '{threshold_file}'. Run the 'threshold' command first.")
    threshold_info = json.loads(threshold_file.read_text())
    threshold = float(threshold_info["threshold"])
    # score with the same method the threshold was chosen for
    method = threshold_info.get("score_method", conf.score_method)

    # no load_and_clean here: every input row should get a prediction, duplicates included
    df = pd.read_csv(input_path)
    X, y = Data(conf).build_features(df, conf.time_mode)
    x = torch.from_numpy(scaler.transform(X[feature_names]).astype(np.float32))

    torch.manual_seed(conf.seed)
    model.eval()
    with torch.no_grad():
        scores = SCORERS[method](model, x).numpy()
        x_hat, _, _ = model(x, sample=False)
        feature_errors = torch.pow(x - x_hat, 2).numpy()

    # per row: the features the model reconstructed worst, i.e. why it looks anomalous
    worst = np.argsort(feature_errors, axis=1)[:, ::-1][:, :top_features]
    explanations = [", ".join(feature_names[i] for i in row) for row in worst]

    out = df.copy()
    out["anomaly_score"] = scores
    out["flagged"] = (scores >= threshold).astype(int)
    out["top_features"] = explanations

    out_path = Path(output_path) if output_path else run_dir / "predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    n_flagged = int(out["flagged"].sum())
    print(f"\n{'=' * 25} Prediction Summary {'=' * 25}")
    print(f"Input:      {input_path} ({len(out):,} rows)")
    print(f"Score:      {method} | threshold {threshold:.2f} ({threshold_info.get('method')})")
    print(f"Flagged:    {n_flagged:,} ({n_flagged / len(out):.3%})")

    if y is not None:
        m = metrics_at_threshold(scores, y, threshold)
        print(f"Labels found -> recall {m['recall']:.3f} | precision {m['precision']:.3f} | "
              f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")

    print("Most anomalous rows:")
    cols = ["anomaly_score", "flagged", "top_features"] + (["Class"] if "Class" in out.columns else [])
    print(out.sort_values("anomaly_score", ascending=False)[cols].head(10).to_string(float_format="%.2f"))
    print(f"Saved to:   {out_path}")
    print(f"{'=' * 70}\n")

    return out
