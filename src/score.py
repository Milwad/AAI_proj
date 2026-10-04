import torch
import math
import numpy as np
from vae import VAE, reconstruction_error, kl_divergence
from typing import Callable
from torch.utils.data import DataLoader

def reconstruction_error_det(model: VAE, x: torch.Tensor) -> torch.Tensor:

    x_hat, _, _ = model(x, sample=False)
    return reconstruction_error(x=x, x_hat=x_hat)

def _sampled_errors(model: VAE, x: torch.Tensor ,n_samples: int = 10) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    errors = []
    mean, logvar = model.encoder(x)
    for k in range(n_samples):
        z = model.reparameterize(mean=mean, logvar=logvar)
        x_hat = model.decoder(z)
        errors.append(reconstruction_error(x=x,x_hat=x_hat))

    return torch.stack(errors, dim=0), mean, logvar

def recon_monte_carlo(model: VAE, x: torch.Tensor, n_samples: int = 10)->torch.Tensor:

    errors, _, _ = _sampled_errors(model=model, x=x, n_samples=n_samples)

    return errors.mean(dim=0)

def negative_elbo(model: VAE, x: torch.Tensor, n_samples: int = 10, beta: float = 1.0)->torch.Tensor:

    errors, mean, logvar = _sampled_errors(model=model, x=x, n_samples=n_samples)
    reconstruction_loss = errors.mean(dim=0)
    kl_loss = kl_divergence(mean=mean, logvar=logvar)

    return reconstruction_loss + beta * kl_loss

# An & Cho 2015
def reconstruction_probability(model: VAE, x: torch.Tensor, n_samples: int = 10) -> torch.Tensor:

    errors, _, _ = _sampled_errors(model=model, x=x, n_samples=n_samples)
    log_k = math.log(n_samples)

    return -(torch.logsumexp(-0.5 * errors, dim=0) - log_k)

def kl(model: VAE, x: torch.Tensor) -> torch.Tensor:

    mean, logvar = model.encoder(x)

    return kl_divergence(mean=mean, logvar=logvar)

SCORERS: dict[str, Callable] = {
    "recon": reconstruction_error_det,
    "recon_mc": recon_monte_carlo,
    "neg_elbo": negative_elbo,
    "recon_prob": reconstruction_probability,
    "kl": kl,
}

def score_loader(model: VAE, loader : DataLoader, method: str, seed: int | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray]:

    if method not in SCORERS:
        raise ValueError(f"Unknown scoring method '{method}'. Available {list(SCORERS.keys())}")

    scorer_func = SCORERS[method]
    model.eval()

    rng_state = torch.get_rng_state() if seed is not None else None
    if seed is not None:
        torch.manual_seed(seed=seed)

    all_scores: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    try:
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:
                    x, y = batch
                    all_y.append(y.cpu().numpy())
                else:
                    x=batch[0]

                score = scorer_func(model, x, **kwargs)
                all_scores.append(score.cpu().numpy())
    finally:
        if rng_state is not None:
            torch.set_rng_state(rng_state)

    scores_arr = np.concatenate(all_scores)
    labels_arr = np.concatenate(all_y) if all_y else np.array([])

    return scores_arr, labels_arr

def per_feature_errors(model: VAE, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:

    model.eval()
    all_errors: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if len(batch)==2:
                x,y = batch
                all_labels.append(y.cpu().numpy())
            else:
                x = batch[0]

            x_hat, _, _ = model(x, sample=False)
            err = torch.pow(x-x_hat, 2)
            all_errors.append(err.cpu().numpy())

    errors_arr = np.concatenate(all_errors, axis=0)
    labels_arr = np.concatenate(all_labels) if all_labels else np.array([])

    return errors_arr, labels_arr

if __name__ == "__main__":
    
    from config import Config
    from data import Data
    from sklearn.metrics import average_precision_score, roc_auc_score
    from vae import load_model

    conf = Config()
    model, ckpt = load_model(f"{conf.results_dir}/{conf.run_name}")

    data = Data(conf)
    X_s, y_s, _, _ = data.get_splits()
    _, val_loader, _ = data.make_dataloaders(X_s, y_s)

    def ap_roc(s, y):
        return average_precision_score(y, s), roc_auc_score(y, s)

    # Shape and alignment: labels come back in the original validation order
    s, y = score_loader(model, val_loader, "recon")
    print(
        f"[1] scores {s.shape}, labels {y.shape} (expect ({len(y_s['val'])},) both) | "
        f"labels in original order: {np.array_equal(y, y_s['val'])} (expect True)"
    )

    # Consistent with training: same score, same model -> same PR-AUC as the best epoch
    ap, roc = ap_roc(s, y)
    print(f"[2] recon AP {ap:.4f} (expect {ckpt['best_value']:.4f}, the best val_ap from training)")

    # Direction: higher = more anomalous
    print(
        f"[3] mean score normal {s[y == 0].mean():.2f} | fraud {s[y == 1].mean():.2f} (expect fraud much higher)"
    )

    # Stochastic scores are reproducible with a seed
    a, _ = score_loader(model, val_loader, "recon_mc", seed=0, n_samples=10)
    b, _ = score_loader(model, val_loader, "recon_mc", seed=0, n_samples=10)
    c, _ = score_loader(model, val_loader, "recon_mc", seed=1, n_samples=10)
    print(
        f"[4] same seed equal: {np.array_equal(a, b)} (expect True) | different seed equal: {np.array_equal(a, c)} (expect False)"
    )

    # neg_elbo = recon_mc + beta * KL (same seed -> same samples)
    e, _ = score_loader(model, val_loader, "neg_elbo", seed=0, n_samples=10, beta=1.0)
    k, _ = score_loader(model, val_loader, "kl")
    print(
        f"[5] neg_elbo == recon_mc + KL: {np.allclose(e, a + k, rtol=1e-5, atol=1e-3)} (expect True) | "
        f"KL min {k.min():.4f} (expect >= 0)"
    )

    # recon_prob with K=1 is exactly half the squared error
    p1, _ = score_loader(model, val_loader, "recon_prob", seed=0, n_samples=1)
    m1, _ = score_loader(model, val_loader, "recon_mc", seed=0, n_samples=1)
    print(
        f"[6] recon_prob(K=1) == 0.5 * recon_mc(K=1): {np.allclose(p1, 0.5 * m1, rtol=1e-5, atol=1e-3)} (expect True)"
    )

    # Method comparison on validation
    print("[7] method comparison on validation:")
    methods = [
        ("recon", {}),
        ("recon_mc", {"n_samples": 10}),
        ("neg_elbo", {"n_samples": 10, "beta": 1.0}),
        ("recon_prob", {"n_samples": 10}),
        ("kl", {}),
    ]
    for name, kw in methods:
        s_m, _ = score_loader(model, val_loader, name, seed=0, **kw)
        ap_m, roc_m = ap_roc(s_m, y)
        print(f"    {name:10s} AP {ap_m:.4f} | ROC-AUC {roc_m:.4f}")
    print(f"    random baseline AP = prevalence = {y.mean():.4f}")
