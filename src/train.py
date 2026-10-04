import torch
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
from vae import VAE
from vae import vae_loss
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim import Adam
from data import Data
from config import Config
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import average_precision_score
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict, replace
from score import reconstruction_error_det

def set_seed(seed: int)->None:
    torch.manual_seed(seed=seed)
    np.random.seed(seed=seed)

def train_one_epoch(model: VAE, loader: DataLoader, optimizer: torch.optim.Optimizer, beta_eff: float, grad_clip: float | None = None) -> dict[str, float]:

    model.train()

    loss_sum: float = 0.0
    reconstruction_loss_sum: float = 0.0
    kl_divergence_sum: float = 0.0
    n: int = 0

    for(x,) in loader:
        x_hat: torch.Tensor
        mean: torch.Tensor
        logvar: torch.Tensor
        x_hat, mean, logvar = model(x=x, sample=True)
        loss: torch.Tensor
        reconstruction_loss: torch.Tensor
        kl_divergence: torch.Tensor
        loss, reconstruction_loss, kl_divergence = vae_loss(x=x, x_hat=x_hat, mean=mean, logvar=logvar, beta=beta_eff)

        optimizer.zero_grad()
        loss.mean().backward()
        if grad_clip:
            clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = x.size(0)
        loss_sum += loss.sum().item()
        reconstruction_loss_sum += reconstruction_loss.sum().item()
        kl_divergence_sum += kl_divergence.sum().item()
        n += batch_size

    return {"loss":loss_sum/n,"recon":reconstruction_loss_sum/n,"kl":kl_divergence_sum/n}

def evaluate(model: VAE, loader: DataLoader, beta: float) -> dict[str, float]:

    model.eval()

    loss_sum: float = 0.0
    reconstruction_loss_sum: float = 0.0
    kl_divergence_sum: float = 0.0
    n_valid: int = 0

    all_scores: list[float] = []
    all_y: list[float] = []

    with torch.no_grad():
        for (x,y) in loader:
            x_hat, mean, logvar = model(x=x, sample=True)
            loss, reconstruction_loss, kl_divergence = vae_loss(x=x, x_hat=x_hat, mean=mean, logvar=logvar, beta=beta)
            # if not fraud
            valid_mask = (y==0)
            if valid_mask.any():                
                loss_sum += loss[valid_mask].sum().item()
                reconstruction_loss_sum += reconstruction_loss[valid_mask].sum().item()
                kl_divergence_sum += kl_divergence[valid_mask].sum().item()
                n_valid += valid_mask.sum().item()

            scores = reconstruction_error_det(model=model,x=x)

            all_scores.extend(scores.cpu().tolist())
            all_y.extend(y.cpu().tolist())

        ap = float(average_precision_score(y_true=all_y, y_score=all_scores))
        n_valid = max(n_valid, 1)

    return {"loss":loss_sum/n_valid,"recon":reconstruction_loss_sum/n_valid,"kl":kl_divergence_sum/n_valid,"ap":ap}

def plot_history(history: list[dict], save_path: Path, best_epoch: int | None = None) -> None:

    epochs = [h["epoch"] for h in history]

    # 3 subplots: [1] Total Loss, [2] Recon vs. KL Trade-off, [3] Validation PR-AUC
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # [1] Total Loss
    ax1.plot(epochs, [h["train_loss"] for h in history], label="Train Loss", color="tab:blue")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="Val Loss (normal)", color="tab:orange")
    ax1.set_title("Total Loss (Recon + beta * KL)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # [2] Warm-up Trade-off: Recon (left axis) vs. KL (right axis)
    color_recon = "tab:blue"
    ax2.plot(epochs, [h["train_recon"] for h in history], label="Train Recon", color=color_recon)
    ax2.plot(epochs, [h["val_recon"] for h in history], label="Val Recon", color="tab:cyan", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Reconstruction Error (MSE)", color=color_recon)
    ax2.tick_params(axis="y", labelcolor=color_recon)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Twin axis for KL Divergence
    ax2_twin = ax2.twinx()
    color_kl = "tab:purple"
    ax2_twin.plot(epochs, [h["train_kl"] for h in history], label="Train KL", color=color_kl)
    ax2_twin.plot(epochs, [h["val_kl"] for h in history], label="Val KL", color="tab:pink", linestyle="--")
    ax2_twin.set_ylabel("KL Divergence", color=color_kl)
    ax2_twin.tick_params(axis="y", labelcolor=color_kl)

    # Combined legend for both axes in panel 2
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    ax2.set_title("Warm-up Trade-off (Recon vs. KL)")

    # [3] Validation PR-AUC (Average Precision)
    val_aps = [h["val_ap"] for h in history]
    ax3.plot(epochs, val_aps, label="Val AP", color="tab:green", linewidth=2)
    if best_epoch is not None:
        ax3.axvline(x=best_epoch, color="tab:red", linestyle=":", label=f"Best (Epoch {best_epoch})")
    ax3.set_title("Validation PR-AUC (Average Precision)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Average Precision")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def fit(conf: Config) -> tuple[VAE, list]:

    set_seed(conf.seed)

    data = Data(conf=conf)
    X_s, y_s, scaler, feature_names = data.get_splits()
    train_loader, val_loader, _ = data.make_dataloaders(X_s=X_s,y_s=y_s)

    model = VAE(input_dim=len(feature_names), hidden_dims=conf.hidden_dims, latent_dim=conf.latent_dim, activation=conf.activation)

    optimizer = Adam(params=model.parameters(), lr=conf.lr)

    if conf.select_by not in ("val_ap", "val_loss"):
        raise ValueError(f"Unknown select_by '{conf.select_by}'. Expected 'val_ap' or 'val_loss'.")
    higher_is_better = (conf.select_by == "val_ap")

    best_value, best_state, best_epoch, bad_epochs = -float("inf") if higher_is_better else float("inf"), None, 0, 0

    history: list[dict] = []

    warmup = conf.kl_warmup_epochs

    for epoch in range(1, conf.max_epochs+1):
        beta_eff = conf.beta * (min(1, epoch/warmup) if warmup > 0 else 1.0)

        tr = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, beta_eff=beta_eff, grad_clip=conf.grad_clip)
        va = evaluate(model=model, loader=val_loader, beta=conf.beta)
        history.append({
            "epoch": epoch,
            "beta_eff":beta_eff,
            "train_loss": tr["loss"],
            "train_recon": tr["recon"],
            "train_kl": tr["kl"],
            "val_loss": va["loss"],
            "val_recon": va["recon"],
            "val_kl": va["kl"],
            "val_ap": va["ap"]
            })
        
        print(
            f"Epoch {epoch:03d}/{conf.max_epochs} | "
            f"beta: {beta_eff:.3f} | "
            f"Train Loss: {tr['loss']:.4f} (Recon: {tr['recon']:.4f}, KL: {tr['kl']:.4f}) | "
            f"Val Loss: {va['loss']:.4f} | "
            f"Val AP: {va['ap']:.4f}"
        )

        if epoch >= warmup:
            current = va["ap"] if higher_is_better else va["loss"]
            improved = (current > best_value) if higher_is_better else (current < best_value)
            if improved:
                best_value = current
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= conf.patience:
                    print(f"Early stopping at epoch {epoch} (best {conf.select_by} was {best_value:.4f} at epoch {best_epoch})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        x_val_norm = torch.from_numpy(X_s["val"][y_s["val"] == 0])
        mu_val, _ = model.encoder(x_val_norm)
        active_units = (mu_val.var(dim=0) > 0.01).sum().item()
        print(f"Active latent units (Var(mu) > 0.01): {active_units}/{conf.latent_dim}")

    run_dir = Path(conf.results_dir) / conf.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    conf_dict = asdict(conf)
    if isinstance(conf_dict.get("activation"), type):
        conf_dict["activation"] = conf_dict["activation"].__name__
    checkpoint = {
        "model_state": model.state_dict(),
        "config": conf_dict,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "feature_names": feature_names,
    }
    torch.save(checkpoint, run_dir / "model.pt")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)
    
    joblib.dump(scaler, run_dir / "scaler.joblib")

    plot_history(history, save_path=run_dir / "curves.png", best_epoch=best_epoch)
    return model, history

def overfit_check(conf: Config) -> None:

    set_seed(conf.seed)
    data = Data(conf=conf)
    X_s, _, _, feature_names = data.get_splits()

    tiny_dataset = TensorDataset(torch.from_numpy(X_s["train"][:512]))
    tiny_loader = DataLoader(tiny_dataset, batch_size=64, shuffle=True)
    model = VAE(
        input_dim=len(feature_names),
        hidden_dims=conf.hidden_dims,
        latent_dim=conf.latent_dim,
        activation=conf.activation,
    )
    optimizer = Adam(params=model.parameters(), lr=conf.lr)
    print("Running overfit sanity check on 512 samples (beta=0.0)...")
    for epoch in range(1, 201):

        stats = train_one_epoch(
            model=model,
            loader=tiny_loader,
            optimizer=optimizer,
            beta_eff=0.0,
            grad_clip=conf.grad_clip,
        )
        if epoch in (1, 50, 100, 200):
            print(f"[overfit] epoch {epoch:3d}: recon {stats['recon']:.3f}")

if __name__ == "__main__":
    
    # overfit_check(Config())
    fit(Config())