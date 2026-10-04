import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import Config
from data import Data
from vae import VAE, load_model
from score import SCORERS

PLOT_FEATURES: list[str] = ["V1", "V2", "V3", "V4", "V10", "V12", "V14", "Amount"]

def sample_transactions(model: VAE, latent_dim: int, n_samples: int, seed: int) -> torch.Tensor:

    # z ~ N(0, I) is the prior the KL term pulls the encoder towards, so decoding it gives new "normal" transactions
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(n_samples, latent_dim, generator=generator)
    model.eval()
    with torch.no_grad():
        # the decoder outputs the mean of p(x|z), in the scaled feature space
        return model.decoder(z)

def to_original_units(x_scaled: np.ndarray, scaler: StandardScaler, feature_names: list[str]) -> pd.DataFrame:

    df = pd.DataFrame(scaler.inverse_transform(x_scaled), columns=feature_names)
    # Amount was stored as log1p(Amount), expm1 undoes that; a decoder can output slightly negative values, so clip
    df["Amount"] = np.expm1(df["Amount"]).clip(lower=0)
    return df

def fidelity_table(real: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:

    rows = []
    for col in real.columns:
        # KS statistic = largest gap between the two cumulative distributions (0 = identical, 1 = no overlap)
        ks = ks_2samp(real[col], generated[col]).statistic
        rows.append({
            "feature": col,
            "real_mean": real[col].mean(),
            "gen_mean": generated[col].mean(),
            "real_std": real[col].std(),
            "gen_std": generated[col].std(),
            "ks_stat": ks,
        })
    return pd.DataFrame(rows)

def correlation_gap(real: pd.DataFrame, generated: pd.DataFrame) -> float:

    # mean absolute difference between the two correlation matrices, upper triangle only (matrix is symmetric)
    diff = np.abs(real.corr().to_numpy() - generated.corr().to_numpy())
    upper = np.triu_indices(diff.shape[0], k=1)
    return float(diff[upper].mean())

def plot_marginals(real: pd.DataFrame, generated: pd.DataFrame, features: list[str], save_path: Path) -> None:

    features = [f for f in features if f in real.columns]
    n_cols = 4
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    for ax, feat in zip(axes.flat, features):
        # shared bins over the central 99% of the real data, so a few outliers do not squash the plot
        low, high = np.percentile(real[feat], [0.5, 99.5])
        bins = np.linspace(low, high, 50)
        ax.hist(real[feat], bins=bins, density=True, alpha=0.5, label="Real (train)", color="tab:blue")
        ax.hist(generated[feat], bins=bins, density=True, alpha=0.5, label="Generated", color="tab:green")
        ax.set_title(feat)
        ax.grid(True, alpha=0.3)

    # hide unused subplots if the feature count is not a multiple of n_cols
    for ax in list(axes.flat)[len(features):]:
        ax.axis("off")

    axes.flat[0].legend()
    fig.suptitle("Real vs. Generated Feature Distributions")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_latent_space(model: VAE, x: np.ndarray, labels: np.ndarray, save_path: Path) -> None:

    with torch.no_grad():
        mean, _ = model.encoder(torch.from_numpy(x))
    mean = mean.numpy()

    # project to 2D if the latent space is bigger than 2
    if mean.shape[1] > 2:
        pca = PCA(n_components=2)
        points = pca.fit_transform(mean)
        var = pca.explained_variance_ratio_
        x_label, y_label = f"PC1 ({var[0]:.0%} var.)", f"PC2 ({var[1]:.0%} var.)"
    else:
        points = mean
        x_label, y_label = "z1", "z2"

    fig, ax = plt.subplots(figsize=(7, 6))
    normal, fraud = labels == 0, labels == 1
    ax.scatter(points[normal, 0], points[normal, 1], s=2, alpha=0.2, label="Normal", color="tab:blue")
    # fraud drawn last so it is on top
    ax.scatter(points[fraud, 0], points[fraud, 1], s=10, alpha=0.8, label="Fraud", color="tab:orange")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Latent Space (encoder means, validation set)")
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def generate(conf: Config, n_samples: int = 5000) -> pd.DataFrame:

    run_dir = Path(conf.results_dir) / conf.run_name
    out_dir = run_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_model(run_dir)
    data = Data(conf)
    X_s, y_s, scaler, feature_names = data.get_splits()

    x_gen = sample_transactions(model, checkpoint["config"]["latent_dim"], n_samples, conf.seed)
    gen_df = to_original_units(x_gen.numpy(), scaler, feature_names)
    real_df = to_original_units(X_s["train"], scaler, feature_names)
    gen_df.to_csv(out_dir / "samples.csv", index=False)

    fidelity = fidelity_table(real_df, gen_df)
    fidelity.to_csv(out_dir / "fidelity.csv", index=False)
    corr_gap = correlation_gap(real_df, gen_df)

    plot_marginals(real_df, gen_df, PLOT_FEATURES, out_dir / "marginals.png")
    plot_latent_space(model, X_s["val"], y_s["val"], out_dir / "latent_space.png")

    # does the detector itself consider the generated samples normal?
    scorer = SCORERS[conf.score_method]
    x_val_normal = torch.from_numpy(X_s["val"][y_s["val"] == 0])
    torch.manual_seed(conf.seed)
    with torch.no_grad():
        gen_scores = scorer(model, x_gen).numpy()
        real_scores = scorer(model, x_val_normal).numpy()

    print(f"\n{'=' * 25} Generation Summary {'=' * 25}")
    print(f"Samples:            {n_samples:,} -> {out_dir / 'samples.csv'}")
    print(f"Mean KS statistic:  {fidelity['ks_stat'].mean():.3f} (0 = identical marginals)")
    print(f"Std ratio gen/real: {(fidelity['gen_std'] / fidelity['real_std']).mean():.3f} (1 = same spread)")
    print(f"Correlation gap:    {corr_gap:.3f} (mean |corr_real - corr_gen|)")
    print(f"Median score ({conf.score_method}): generated {np.median(gen_scores):.2f} | real normal (val) {np.median(real_scores):.2f}")

    threshold_file = run_dir / "threshold.json"
    if threshold_file.exists():
        threshold = float(json.loads(threshold_file.read_text())["threshold"])
        print(f"Flagged as fraud:   {np.mean(gen_scores >= threshold):.2%} of generated samples (threshold {threshold:.2f})")

    print(f"Worst-matched features (KS):")
    print(fidelity.sort_values("ks_stat", ascending=False).head(5).to_string(index=False, float_format="%.3f"))
    print(f"{'=' * 70}\n")

    return gen_df
