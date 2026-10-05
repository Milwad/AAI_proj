import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import replace
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from config import Config
from data import Data
from vae import load_model
from score import SCORERS, score_loader
from train import fit

EXPERIMENT_DIR: str = "experiments"
# one-factor-at-a-time ablation around the baseline (beta=1, latent_dim=8); beta=0 is a plain autoencoder
BETAS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
LATENT_DIMS: tuple[int, ...] = (2, 4, 8, 16, 32)
# looked interesting after ablation study
EXTRA_CONFIGS: dict[str, dict] = {
    "beta4.0_lat16": {"beta": 4.0, "latent_dim": 16},
}

def experiment_grid(base: Config) -> dict[str, dict]:

    # dict keyed by name, so the shared baseline config (beta=1, latent=8) only appears once
    grid: dict[str, dict] = {}
    for beta in BETAS:
        grid[f"beta{beta}_lat{base.latent_dim}"] = {"beta": beta, "latent_dim": base.latent_dim}
    for latent_dim in LATENT_DIMS:
        grid[f"beta{base.beta}_lat{latent_dim}"] = {"beta": base.beta, "latent_dim": latent_dim}

    grid.update(EXTRA_CONFIGS)
    return grid

def train_grid(base: Config, grid: dict[str, dict], seeds: list[int]) -> None:

    total = len(grid) * len(seeds)
    i = 0
    for name, overrides in grid.items():
        for seed in seeds:
            i += 1
            run_name = f"{EXPERIMENT_DIR}/{name}_s{seed}"
            # skip finished runs, so an interrupted experiment can simply be restarted
            if (Path(base.results_dir) / run_name / "model.pt").exists():
                print(f"[run {i}/{total}] {run_name}: already trained, skipping")
                continue
            print(f"\n[run {i}/{total}] {run_name}: {overrides}, seed {seed}")
            start = time.perf_counter()
            fit(replace(base, run_name=run_name, seed=seed, **overrides))
            print(f"[run {i}/{total}] finished in {time.perf_counter() - start:.0f}s")

def metric_row(model: str, seed: int, split: str, score: str, scores: np.ndarray, labels: np.ndarray) -> dict:

    return {
        "model": model,
        "seed": seed,
        "split": split,
        "score": score,
        "ap": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }

def score_vae_runs(base: Config, grid: dict[str, dict], seeds: list[int], loaders: dict) -> list[dict]:

    rows: list[dict] = []
    for name in grid:
        for seed in seeds:
            run_dir = Path(base.results_dir) / EXPERIMENT_DIR / f"{name}_s{seed}"
            model, _ = load_model(run_dir)
            for split, loader in loaders.items():
                for method in SCORERS:
                    scores, labels = score_loader(model, loader, method, seed=seed)
                    rows.append(metric_row(f"VAE {name}", seed, split, method, scores, labels))
        print(f"[scored] {name}")
    return rows

def score_baselines(base: Config, X_s: dict, y_s: dict, seeds: list[int]) -> list[dict]:

    rows: list[dict] = []

    # linear counterpart of the (V)AE: same bottleneck size, score = squared reconstruction error
    pca = PCA(n_components=base.latent_dim).fit(X_s["train"])
    for split in ("val", "test"):
        x = X_s[split]
        x_hat = pca.inverse_transform(pca.transform(x))
        scores = np.sum((x - x_hat) ** 2, axis=1)
        # PCA is deterministic, so every seed gives the same row; repeated so the tables average the same way
        for seed in seeds:
            rows.append(metric_row(f"PCA (k={base.latent_dim})", seed, split, "recon", scores, y_s[split]))

    # non-generative baseline: isolation forest, also trained on normal transactions only
    for seed in seeds:
        forest = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(X_s["train"])
        for split in ("val", "test"):
            # score_samples is high for normal points, so negate it: higher = more anomalous, like our scores
            scores = -forest.score_samples(X_s[split])
            rows.append(metric_row("Isolation Forest", seed, split, "iforest", scores, y_s[split]))

    print("[scored] baselines")
    return rows

def summarise(results: pd.DataFrame) -> pd.DataFrame:

    # mean and standard deviation over seeds, one row per model / split / score
    summary = results.groupby(["model", "split", "score"])[["ap", "roc_auc"]].agg(["mean", "std"])
    # flatten the two-level column names, e.g. ("ap", "mean") -> "ap_mean"
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()

def comparison_table(summary: pd.DataFrame, score: str) -> pd.DataFrame:

    # one row per model: val AP, test AP and test ROC-AUC, all as "mean +/- std"
    def fmt(df: pd.DataFrame, metric: str) -> pd.Series:
        return df.apply(lambda r: f"{r[f'{metric}_mean']:.3f} +/- {r[f'{metric}_std']:.3f}", axis=1)

    vae = summary[(summary["score"] == score) & summary["model"].str.startswith("VAE")]
    baselines = summary[~summary["model"].str.startswith("VAE")]
    rows = pd.concat([vae, baselines])

    val = rows[rows["split"] == "val"].set_index("model")
    test = rows[rows["split"] == "test"].set_index("model")
    table = pd.DataFrame({
        "val_ap": fmt(val, "ap"),
        "test_ap": fmt(test, "ap"),
        "test_roc_auc": fmt(test, "roc_auc"),
        "_sort": val["ap_mean"],
    })
    return table.sort_values("_sort", ascending=False).drop(columns="_sort")

def plot_ablation(summary: pd.DataFrame, base: Config, save_path: Path) -> None:

    recon = summary[summary["score"] == "recon"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, values, key, label in (
        (ax1, BETAS, lambda v: f"VAE beta{v}_lat{base.latent_dim}", "beta (KL weight)"),
        (ax2, LATENT_DIMS, lambda v: f"VAE beta{base.beta}_lat{v}", "latent dimension"),
    ):
        for split, color in (("val", "tab:blue"), ("test", "tab:orange")):
            part = recon[recon["split"] == split].set_index("model")
            means = [part.loc[key(v), "ap_mean"] for v in values]
            stds = [part.loc[key(v), "ap_std"] for v in values]
            # categorical x positions, so beta=0 and the unevenly spaced values are all readable
            ax.errorbar(range(len(values)), means, yerr=stds, marker="o", capsize=4, label=split, color=color)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(label)
        ax.set_ylabel("PR-AUC (AP)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax1.set_title(f"Effect of beta (latent_dim = {base.latent_dim})")
    ax2.set_title(f"Effect of latent size (beta = {base.beta})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def run_experiments(base: Config, seeds: list[int], skip_train: bool = False) -> pd.DataFrame:

    out_dir = Path(base.results_dir) / EXPERIMENT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = experiment_grid(base)

    if not skip_train:
        train_grid(base, grid, seeds)

    # load the data once and share it between all runs
    data = Data(base)
    X_s, y_s, _, _ = data.get_splits()
    _, val_loader, test_loader = data.make_dataloaders(X_s, y_s)
    loaders = {"val": val_loader, "test": test_loader}

    rows = score_vae_runs(base, grid, seeds, loaders) + score_baselines(base, X_s, y_s, seeds)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "all_runs.csv", index=False)

    summary = summarise(results)
    summary.to_csv(out_dir / "summary.csv", index=False)
    plot_ablation(summary, base, out_dir / "ablation.png")

    # table 1: models compared with the default score, sorted by validation AP
    table = comparison_table(summary, score=base.score_method)
    table.to_csv(out_dir / "model_comparison.csv")

    # table 2: every scoring method for the baseline configuration
    base_name = f"VAE beta{base.beta}_lat{base.latent_dim}"
    scores = summary[summary["model"] == base_name].pivot(index="score", columns="split", values="ap_mean")
    scores = scores.sort_values("val", ascending=False)
    scores.to_csv(out_dir / "score_comparison.csv")

    best_model = table.index[0]
    print(f"\n{'=' * 25} Experiment Summary ({len(seeds)} seeds) {'=' * 25}")
    print(f"Model comparison (score '{base.score_method}' for VAEs, sorted by validation AP):")
    print(table.to_string())
    print(f"\nSelected on validation: {best_model} -> test AP {table.loc[best_model, 'test_ap']}")
    print(f"\nScoring methods for {base_name} (mean AP over seeds):")
    print(scores.to_string(float_format="%.4f"))
    print(f"\nSaved: {out_dir / 'all_runs.csv'}, summary.csv, model_comparison.csv, score_comparison.csv, ablation.png")
    print(f"{'=' * 75}\n")

    return results
