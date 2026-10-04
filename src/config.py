from dataclasses import dataclass
import torch

DATA_PATH = "data/creditcard.csv"
#DATA_PATH = "data/creditcard_2023.csv"
RESULTS_DIR = "results"

@dataclass
class Config:

    data_path: str = DATA_PATH
    results_dir: str = RESULTS_DIR
    # either drop or cyclic
    time_mode: str = "drop"
    normal_split: tuple = (0.6, 0.2, 0.2)
    fraud_val_split: float = 0.5
    batch_size: int = 256
    latent_dim: int = 8
    # enc uses this normally, dec uses it reversed
    hidden_dims: tuple = (64, 32)
    activation: type = torch.nn.ReLU
    split_seed: int = 1
    seed : int = 1
    beta : float = 1.0
    lr: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
    kl_warmup_epochs: int = 10
    grad_clip: float | None = 10.0
    select_by: str = "val_ap"
    run_name: str = "baseline"
    score_method: str = "recon"
    score_samples: int = 10
    threshold_method: str = "max_f1"
    threshold_percentile: float = 99.9
    recall_target: float = 0.8
    fp_cost: float = 5.0