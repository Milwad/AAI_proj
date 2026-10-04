import numpy as np
import json
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, roc_curve
from typing import Callable, Any
from pathlib import Path
from config import Config
from score import per_feature_errors, score_loader
from vae import VAE, load_model
from data import Data

REAL_PREVALENCE: float = 492 / 284_807

def metrics_at_threshold(scores: np.ndarray, labels:np.ndarray, threshold: float, prevalence=None) -> dict[str, Any]:

    pred = (scores >= threshold)

    tp = int(np.sum((pred==1)&(labels==1)))
    fp = int(np.sum((pred==1)&(labels==0)))
    tn = int(np.sum((pred==0)&(labels==0)))
    fn = int(np.sum((pred==0)&(labels==1)))

    precision = float(tp/(tp+fp)) if (tp+fp) > 0 else 0.0
    recall = float(tp/(tp+fn)) if (tp+fn) > 0 else 0.0
    f1 = float(2*precision*recall/(precision+recall)) if (precision+recall) > 0 else 0.0
    fpr = float(fp/(fp+tn)) if (fp+tn) > 0 else 0.0

    if prevalence is not None:
        denominator = recall * prevalence + fpr * (1.0-prevalence)
        precision_adj = float((recall*prevalence)/denominator) if denominator > 0 else 0.0
    else:
        precision_adj = None

    return {
        "tp":tp,
        "fp":fp,
        "tn":tn,
        "fn":fn,
        "precision":precision,
        "recall":recall,
        "f1":f1,
        "fpr":fpr,
        "precision_adj":precision_adj
    }

def threshold_percentile(scores: np.ndarray, labels: np.ndarray, percentile: float = 99.9) -> float:

    normal_scores = scores[labels==0]
    return float(np.percentile(normal_scores, percentile))

def threshold_max_f1(scores: np.ndarray, labels: np.ndarray) -> float:

    P, R, T = precision_recall_curve(labels, scores)
    P_m = P[:-1]
    R_m = R[:-1]

    denominator = P_m + R_m
    f1_scores = np.where(denominator > 0, (2*P_m*R_m)/denominator, 0.0)

    best_index = int(np.argmax(f1_scores))
    return float(T[best_index])

def threshold_recall_target(scores: np.ndarray, labels: np.ndarray, target: float = 0.8) -> float:

    P, R, T = precision_recall_curve(labels, scores)
    R_m = R[:-1] 

    qualifying = np.where(R_m >= target)[0]
    if len(qualifying) == 0:
        raise ValueError(f"Unreachable recall target {target}. Max achievable recall is {R_m.max():.4f}.")

    best_index = int(qualifying[-1])
    return float(T[best_index])

THRESHOLDS: dict[str, Callable] = {
    "percentile": threshold_percentile,
    "max_f1": threshold_max_f1,
    "recall_target": threshold_recall_target,
}

def select_threshold(scores: np.ndarray, labels: np.ndarray, method: str, **params)->float:

    if method not in THRESHOLDS:
        raise ValueError(f"Unknown threshold method '{method}'. Available: {list(THRESHOLDS.keys())}")

    return THRESHOLDS[method](scores, labels, **params)

def save_threshold(run_dir: str | Path, method: str, params: dict, threshold: float, val_metrics: dict)->Path:

    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir/"threshold.json"

    def _clean_val(v):
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return float(v)
        return v

    data = {
        "method": method,
        "params": params,
        "threshold": float(threshold),
        "val_metrics": {k: _clean_val(v) for k, v in val_metrics.items()}
    }
    with open(out_file, "w") as f:
        json.dump(data, f, indent=4)

    return out_file

def choose_threshold(conf: Config) -> tuple[float, dict]:

    model, checkpoint = load_model(f"{conf.results_dir}/{conf.run_name}")
    data = Data(conf=conf)
    X_s, y_s, _, _ = data.get_splits()
    _, val_loader, _ = data.make_dataloaders(X_s=X_s, y_s=y_s)

    scores, labels = score_loader(model=model, loader=val_loader, method=conf.score_method)
    params = {}
    if conf.threshold_method == "percentile":
        params["percentile"] = conf.threshold_percentile
    elif conf.threshold_method == "recall_target":
        params["target"] = conf.recall_target

    threshold = select_threshold(scores=scores, labels=labels, method=conf.threshold_method,**params)
    val_metrics = metrics_at_threshold(scores = scores, labels=labels, threshold=threshold, prevalence=REAL_PREVALENCE)

    run_dir = Path(conf.results_dir)/conf.run_name
    save_threshold(run_dir=run_dir, method=conf.threshold_method, params=params, threshold=threshold, val_metrics=val_metrics)

    print(f"\n[Saved] Threshold = {threshold:.4f} ({conf.threshold_method}) saved to {run_dir/'threshold.json'}")

    return threshold, val_metrics

def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:

    if k <= 0:
        return 0.0
    # double slice -1 reverses list
    order = np.argsort(scores)[::-1]
    top = labels[order[:k]]
    return float(top.mean())

def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:

    total_pos = int(labels.sum())
    if total_pos == 0:
        return 0.0
    # double slice -1 reverses list
    order = np.argsort(scores)[::-1]
    top = labels[order[:k]]
    return float(top.sum()/total_pos)

def stratified_bootstrap_confidence_interval(scores: np.ndarray, labels: np.ndarray, metric_func: Callable[[np.ndarray, np.ndarray], float], n_boot: int = 1000, seed: int = 0)-> tuple[float, float]:

    rng = np.random.default_rng(seed=seed)
    fraud_index = np.where(labels == 1)[0]
    valid_index = np.where(labels == 0)[0]
    n_fraud = len(fraud_index)
    n_valid = len(valid_index)
    
    values = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample_fraud = rng.choice(fraud_index, size=n_fraud, replace=True)
        sample_valid = rng.choice(valid_index, size=n_valid, replace=True)
        index = np.concatenate([sample_fraud, sample_valid])
        values[b] = metric_func(scores[index], labels[index])

    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

def plot_pr_curve(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    test_scores: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    threshold: float | None = None,
    save_path: str | Path | None = None
) -> None:
    
    """Plots PR curve with random baseline and operating point."""
    fig, ax = plt.subplots(figsize=(7, 5))

    p_val, r_val, _ = precision_recall_curve(val_labels, val_scores)
    ap_val = average_precision_score(val_labels, val_scores)
    ax.plot(r_val, p_val, label=f"Validation (AP = {ap_val:.4f})", color="tab:blue", lw=2)

    if test_scores is not None and test_labels is not None:
        p_test, r_test, _ = precision_recall_curve(test_labels, test_scores)
        ap_test = average_precision_score(test_labels, test_scores)
        ax.plot(r_test, p_test, label=f"Test (AP = {ap_test:.4f})", color="tab:orange", lw=2)
        baseline = float(test_labels.mean())
        ax.axhline(baseline, color="gray", linestyle="--", label=f"Random baseline ({baseline:.4f})")
    else:
        baseline = float(val_labels.mean())
        ax.axhline(baseline, color="gray", linestyle="--", label=f"Random baseline ({baseline:.4f})")

    active_scores = test_scores if test_scores is not None else val_scores
    active_labels = test_labels if test_labels is not None else val_labels
    if threshold is not None:
        m = metrics_at_threshold(active_scores, active_labels, threshold)
        ax.plot(
            m["recall"],
            m["precision"],
            "ro",
            markersize=8,
            label=f"Threshold {threshold:.2f} (R={m['recall']:.3f}, P={m['precision']:.3f})",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_score_histogram(scores: np.ndarray, labels: np.ndarray, threshold: float, save_path: str | Path | None = None,) -> None:

    fig, ax = plt.subplots(figsize=(7, 5))
    min_val = max(float(scores.min()), 0.1)
    max_val = float(scores.max())
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 60)

    norm_scores = scores[labels == 0]
    fraud_scores = scores[labels == 1]

    norm_weights = np.ones_like(norm_scores) / len(norm_scores)
    fraud_weights = np.ones_like(fraud_scores) / len(fraud_scores)

    ax.hist(norm_scores, bins=bins, weights=norm_weights, alpha=0.6, label="Normal", color="tab:blue")
    ax.hist(fraud_scores, bins=bins, weights=fraud_weights, alpha=0.6, label="Fraud", color="tab:orange")
    ax.axvline(threshold, color="red", linestyle="--", lw=2, label=f"Threshold ({threshold:.2f})")

    ax.set_xscale("log")
    ax.set_xlabel("Anomaly Score (Reconstruction Error, log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: Normal vs. Fraud")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_confusion_matrix(metrics: dict[str, Any], threshold: float, save_path: str | Path | None = None,) -> None:

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Normal", "Pred Fraud"])
    ax.set_yticklabels(["Actual Normal", "Actual Fraud"])

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    ax.set_title(f"Confusion Matrix (t = {threshold:.2f})")
    fig.colorbar(im)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_feature_importance(feature_names: list[str], errors: np.ndarray, labels: np.ndarray, top_n: int = 15, save_path: str | Path | None = None,) -> None:

    normal_err = errors[labels == 0].mean(axis=0)
    fraud_err = errors[labels == 1].mean(axis=0)
    ratio = fraud_err / np.maximum(normal_err, 1e-12)

    sort_idx = np.argsort(ratio)[::-1][:top_n]
    top_names = [feature_names[i] for i in sort_idx]
    top_ratios = ratio[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(len(sort_idx)), top_ratios[::-1], color="tab:purple")
    ax.set_yticks(range(len(sort_idx)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Reconstruction Error Ratio (Fraud / Normal)")
    ax.set_title(f"Top {top_n} Anomaly Features by Reconstruction Error Ratio")

    max_val = max(top_ratios)
    for bar, val in zip(bars, top_ratios[::-1]):
        ax.text(val + max_val * 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.1f}x", va="center", fontsize=9)

    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)

def run_evaluation(conf: Config, split: str = "val", force: bool = False) -> dict[str, Any]:

    run_dir = Path(conf.results_dir) / conf.run_name
    figs_dir = run_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    if split == "test":
        test_file = run_dir / "test_metrics.json"
        if test_file.exists() and not force:
            raise RuntimeError(
                f"'{test_file}' already exists. Use force=True to overwrite. "
                "Follow the 'touch test set once' rule!"
            )

    thresh_file = run_dir / "threshold.json"
    if not thresh_file.exists():
        raise FileNotFoundError(f"Threshold file not found at '{thresh_file}'. Run choose_threshold first.")
    with open(thresh_file, "r") as f:
        threshold_info = json.load(f)
    threshold = float(threshold_info["threshold"])

    model, ckpt = load_model(run_dir)
    data = Data(conf)
    X_s, y_s, _, feature_names = data.get_splits()
    _, val_loader, test_loader = data.make_dataloaders(X_s, y_s)

    val_scores, val_labels = score_loader(model, val_loader, conf.score_method, seed=conf.seed)

    if split == "val":
        scores, labels = val_scores, val_labels
        active_loader = val_loader
    elif split == "test":
        scores, labels = score_loader(model, test_loader, conf.score_method, seed=conf.seed)
        active_loader = test_loader
    else:
        raise ValueError(f"Unknown split '{split}'. Must be 'val' or 'test'.")

    # Metrics
    n = int(len(labels))
    n_fraud = int(labels.sum())
    prevalence = float(labels.mean())
    ap = float(average_precision_score(labels, scores))
    roc_auc = float(roc_auc_score(labels, scores))

    # Bootstrap CIs (1,000 resamples)
    ap_ci = stratified_bootstrap_confidence_interval(scores, labels, lambda s, y: float(average_precision_score(y, s)), n_boot=1000, seed=conf.seed)
    roc_auc_ci = stratified_bootstrap_confidence_interval(scores, labels, lambda s, y: float(roc_auc_score(y, s)), n_boot=1000, seed=conf.seed)

    # Precision & recall @ k
    k_values = [50, 100, 200]
    p_at_k = {k: round(precision_at_k(scores, labels, k), 3) for k in k_values}
    r_at_k = {k: round(recall_at_k(scores, labels, k), 3) for k in k_values}

    # Primary threshold metrics & CIs at fixed threshold
    primary = metrics_at_threshold(scores, labels, threshold, prevalence=REAL_PREVALENCE)
    primary["threshold"] = threshold
    primary["recall_ci"] = list(
        stratified_bootstrap_confidence_interval(
            scores, labels,
            lambda s, y: metrics_at_threshold(s, y, threshold, prevalence=REAL_PREVALENCE)["recall"],
            n_boot=1000, seed=conf.seed,
        )
    )
    primary["precision_ci"] = list(
        stratified_bootstrap_confidence_interval(
            scores, labels,
            lambda s, y: metrics_at_threshold(s, y, threshold, prevalence=REAL_PREVALENCE)["precision"],
            n_boot=1000, seed=conf.seed,
        )
    )

    # Secondary thresholds (fitted on val, evaluated on current split)
    sec_configs = [
        ("percentile_99_0", "percentile", {"percentile": 99.0}),
        ("percentile_99_9", "percentile", {"percentile": 99.9}),
        ("max_f1", "max_f1", {}),
        ("recall_80", "recall_target", {"target": 0.8}),
    ]
    secondary = {}
    for name, method, params in sec_configs:
        t_sec = select_threshold(val_scores, val_labels, method, **params)
        m_sec = metrics_at_threshold(scores, labels, t_sec, prevalence=REAL_PREVALENCE)
        m_sec["threshold"] = float(t_sec)
        secondary[name] = m_sec

    results: dict[str, Any] = {
        "split": split,
        "n": n,
        "n_fraud": n_fraud,
        "prevalence": prevalence,
        "ap": ap,
        "ap_ci": list(ap_ci),
        "roc_auc": roc_auc,
        "roc_auc_ci": list(roc_auc_ci),
        "precision_at_k": p_at_k,
        "recall_at_k": r_at_k,
        "primary": primary,
        "secondary": secondary,
    }

    # Generate figures
    if split == "test":
        plot_pr_curve(val_scores, val_labels, test_scores=scores, test_labels=labels,
                      threshold=threshold, save_path=figs_dir / f"{split}_pr_curve.png")
    else:
        plot_pr_curve(val_scores, val_labels, threshold=threshold,
                      save_path=figs_dir / f"{split}_pr_curve.png")

    plot_score_histogram(scores, labels, threshold=threshold, save_path=figs_dir / f"{split}_score_histogram.png")
    plot_confusion_matrix(primary, threshold=threshold, save_path=figs_dir / f"{split}_confusion_matrix.png")

    err_feat, y_feat = per_feature_errors(model, active_loader)
    plot_feature_importance(feature_names, err_feat, y_feat, top_n=15, save_path=figs_dir / f"{split}_feature_importance.png")

    # If test split, save to test_metrics.json
    if split == "test":
        test_file = run_dir / "test_metrics.json"

        def _clean_json(obj: Any) -> Any:
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (list, tuple)):
                return [_clean_json(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _clean_json(v) for k, v in obj.items()}
            return obj

        with open(test_file, "w") as f:
            json.dump(_clean_json(results), f, indent=4)
        print(f"\n[Saved] Test metrics successfully written to '{test_file}'")

    # Formatted summary printout
    print(f"\n{'='*25} Evaluation Summary ({split.upper()}) {'='*25}")
    print(f"Samples:     {n:,} | Fraud: {n_fraud:,} | Prevalence: {prevalence:.4%}")
    print(f"PR-AUC (AP): {ap:.4f}  (95% CI: [{ap_ci[0]:.4f}, {ap_ci[1]:.4f}])")
    print(f"ROC-AUC:     {roc_auc:.4f}  (95% CI: [{roc_auc_ci[0]:.4f}, {roc_auc_ci[1]:.4f}])")
    print(f"Precision@k: {p_at_k}")
    print(f"Recall@k:    {r_at_k}")
    print(f"Threshold:   {threshold:.4f} ({threshold_info.get('method')})")
    print(f"Confusion:   TP={primary['tp']}, FP={primary['fp']}, FN={primary['fn']}, TN={primary['tn']}")
    print(f"Recall:      {primary['recall']:.4f}  (95% CI: [{primary['recall_ci'][0]:.4f}, {primary['recall_ci'][1]:.4f}])")
    print(f"Precision:   {primary['precision']:.4f} (real prev: {primary['precision_adj']:.4f})  (95% CI: [{primary['precision_ci'][0]:.4f}, {primary['precision_ci'][1]:.4f}])")
    print(f"F1 score:    {primary['f1']:.4f}")
    print(f"FPR:         {primary['fpr']:.4f}")
    print(f"{'='*68}\n")

    return results

# if __name__ == "__main__":
#     from config import Config
#     from data import Data
#     from score import score_loader
#     from vae import load_model

#     conf = Config()
#     model, ckpt = load_model(f"{conf.results_dir}/{conf.run_name}")
#     data = Data(conf)
#     X_s, y_s, _, _ = data.get_splits()
#     _, val_loader, _ = data.make_dataloaders(X_s, y_s)
#     s, y = score_loader(model, val_loader, conf.score_method)
#     REAL_PREVALENCE = 492 / 284_807  # fraud rate in the full original dataset

#     # [1] Percentile threshold controls the false-positive rate
#     for q in (99.0, 99.9):
#         t = select_threshold(s, y, "percentile", percentile=q)
#         m = metrics_at_threshold(s, y, t)
#         print(
#             f"[1] p{q}: t={t:.2f} | FPR {m['fpr']:.4f} (expect ~{1 - q / 100:.4f}) | "
#             f"recall {m['recall']:.3f} | precision {m['precision']:.3f}"
#         )

#     # [2] Confusion-matrix bookkeeping
#     print(
#         f"[2] TP+FP+FN+TN = {m['tp'] + m['fp'] + m['fn'] + m['tn']:,} (expect {len(y):,}) | "
#         f"TP+FN = {m['tp'] + m['fn']} (expect {int(y.sum())})"
#     )

#     # [3] Max-F1: no nearby threshold does better
#     t_f1 = select_threshold(s, y, "max_f1")
#     f1 = metrics_at_threshold(s, y, t_f1)["f1"]
#     nearby = [metrics_at_threshold(s, y, t_f1 * f)["f1"] for f in (0.9, 0.95, 1.05, 1.1)]
#     print(
#         f"[3] max-F1: t={t_f1:.2f}, F1 {f1:.3f} | nearby F1s {np.round(nearby, 3)} (expect all <= {f1:.3f})"
#     )

#     # [4] Recall target: reaches the target, and is the strictest threshold that does
#     t_r = select_threshold(s, y, "recall_target", target=0.8)
#     r = metrics_at_threshold(s, y, t_r)["recall"]
#     next_stricter = np.min(s[s > t_r])
#     r_next = metrics_at_threshold(s, y, next_stricter)["recall"]
#     print(
#         f"[4] recall target 0.8: t={t_r:.2f}, recall {r:.3f} (expect >= 0.800) | "
#         f"next stricter threshold recall {r_next:.3f} (expect < 0.800)"
#     )

#     # [5] Monotonicity: stricter threshold -> recall and FPR never go up
#     ts = np.percentile(s, [90, 95, 99, 99.5, 99.9])
#     ms = [metrics_at_threshold(s, y, t) for t in ts]
#     rec_ok = all(a["recall"] >= b["recall"] for a, b in zip(ms, ms[1:]))
#     fpr_ok = all(a["fpr"] >= b["fpr"] for a, b in zip(ms, ms[1:]))
#     print(f"[5] recall non-increasing: {rec_ok} | FPR non-increasing: {fpr_ok} (expect True, True)")

#     # [6] Prevalence adjustment: real-world precision is lower than validation precision
#     m = metrics_at_threshold(s, y, t_f1, prevalence=REAL_PREVALENCE)
#     print(
#         f"[6] at max-F1: precision val {m['precision']:.3f} | at real prevalence {m['precision_adj']:.3f} (expect lower)"
#     )

#     # [7] Operating points on validation (the table for your report)
#     print("[7] operating points on validation:")
#     options = [
#         ("percentile", {"percentile": 99.0}),
#         ("percentile", {"percentile": 99.9}),
#         ("max_f1", {}),
#         ("recall_target", {"target": 0.8}),
#     ]
#     for name, kw in options:
#         t = select_threshold(s, y, name, **kw)
#         m = metrics_at_threshold(s, y, t, prevalence=REAL_PREVALENCE)
#         print(
#             f"    {name:13s} {str(kw):22s} t={t:7.2f} | flagged {m['tp'] + m['fp']:4d} | "
#             f"recall {m['recall']:.3f} | precision {m['precision']:.3f} (real {m['precision_adj']:.3f}) | "
#             f"F1 {m['f1']:.3f} | FPR {m['fpr']:.4f}"
#         )

#     # [8] Save threshold
#     choose_threshold(conf)

if __name__ == "__main__":
    
    conf = Config()
    run_dir = Path(conf.results_dir) / conf.run_name
    res = run_evaluation(conf, split="test")
    saved = json.loads((run_dir / "threshold.json").read_text())

    # # for val set ONLY
    # # [1] Threshold-free metrics reproduce the known validation values
    # print(
    #     f"[1] AP {res['ap']:.4f} (expect 0.7633) | ROC-AUC {res['roc_auc']:.4f} (expect 0.9514) | "
    #     f"n {res['n']:,} / fraud {res['n_fraud']} (expect 56,887 / 236)"
    # )

    # # [2] The loaded threshold gives exactly the saved validation confusion matrix
    # same = all(res["primary"][k] == saved["val_metrics"][k] for k in ("tp", "fp", "fn", "tn"))
    # print(f"[2] primary confusion matrix == threshold.json: {same} (expect True)")

    # # [3] Bootstrap intervals contain the point estimates and have a plausible width
    # lo, hi = res["ap_ci"]
    # print(f"[3] AP CI [{lo:.3f}, {hi:.3f}] contains AP: {lo <= res['ap'] <= hi} (expect about [0.71, 0.82], True)")

    # # [4] precision@k at k = #flagged equals precision at the threshold
    # model, _ = load_model(run_dir)
    # data = Data(conf)
    # X_s, y_s, _, _ = data.get_splits()
    # _, val_loader, _ = data.make_dataloaders(X_s, y_s)
    # s, y = score_loader(model, val_loader, conf.score_method)
    # k = res["primary"]["tp"] + res["primary"]["fp"]
    # print(
    #     f"[4] precision@{k} {precision_at_k(s, y, k):.3f} == precision at threshold {res['primary']['precision']:.3f} (expect equal, 0.861)"
    # )

    # # [5] precision@k values
    # print(f"[5] precision@k {res['precision_at_k']} (expect about 50: 0.96, 100: 0.92, 200: 0.865)")