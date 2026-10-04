import numpy as np
import pandas as pd
import sklearn
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import Config
from sklearn.preprocessing import StandardScaler
from joblib import dump
from torch.utils.data import TensorDataset, DataLoader

class Data:

    def __init__(self, conf: Config):
        self.conf = conf

    def load_and_clean(self, path: str) -> pd.DataFrame:

        df : pd.DataFrame = pd.read_csv(path)
        df = df.drop_duplicates().reset_index(drop=True) # fix indices after dropping dupes
        return df
    

    def build_features(self, df : pd.DataFrame, time_mode : str = "drop") -> tuple[pd.DataFrame, np.typing.NDArray | None]:

        # Coloumn 2 until the third last coloumn, a.k.a 2-29
        X : pd.DataFrame = df.loc[:, "V1":"V28"].copy()
        # ln(1+x), avoids negative inf for 0
        X["Amount"] = np.log1p(df["Amount"])
        # if we keep time, we should make it cyclic, so that 23:59 and 00:01 is not on opposite ends in the algos eyes
        if time_mode == "cyclic":
            hour = (df["Time"]/3600) % 24
            X["hour_sin"], X["hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
        y : np.typing.NDArray | None = df["Class"].to_numpy() if "Class" in df.columns else None
        return X, y
    

    def make_split(self, y : np.typing.NDArray, normal_split : tuple[float, float, float], fraud_val_split : float, seed : int) -> dict[str, np.typing.NDArray[np.int64]]:

        normal_index : np.typing.NDArray = np.where(y == 0)[0]
        fraud_index : np.typing.NDArray = np.where(y == 1)[0]
        train_split, val_split, test_split = normal_split
        temp_split = test_split + val_split
        
        train_index : np.typing.NDArray
        temp_index : np.typing.NDArray
        train_index, temp_index = train_test_split(normal_index, test_size=temp_split,random_state=seed,shuffle=True)

        val_test_split = test_split/temp_split

        normal_val_index : np.typing.NDArray
        normal_test_index : np.typing.NDArray 
        normal_val_index, normal_test_index = train_test_split(temp_index, test_size=val_test_split,random_state=seed,shuffle=True)
        
        fraud_val_index : np.typing.NDArray
        fraud_test_index : np.typing.NDArray
        fraud_val_index, fraud_test_index = train_test_split(fraud_index, test_size=1-fraud_val_split, random_state=seed, shuffle=True)

        val_index : np.typing.NDArray = np.concat([normal_val_index, fraud_val_index])
        test_index : np.typing.NDArray = np.concat([normal_test_index, fraud_test_index])

        return {"train":train_index,"val":val_index,"test":test_index}
    

    def get_splits(self) -> tuple[dict[str, np.typing.NDArray[np.float32]], dict[str, np.typing.NDArray[np.int64]], StandardScaler, list[str]]:

        results_dir = Path(self.conf.results_dir)
        split_path : Path = Path(results_dir) / f"splits_seed_{self.conf.split_seed}.npz"

        df : pd.DataFrame = self.load_and_clean(self.conf.data_path)
        # X = inputs
        X :pd.DataFrame
        # y = labels
        y : np.typing.NDArray
        X, y = self.build_features(df, self.conf.time_mode)

        if split_path.is_file():
            with np.load(split_path) as split_file:
                splits = {k: split_file[k] for k in split_file.files}
        else:
            splits = self.make_split(y, self.conf.normal_split, self.conf.fraud_val_split, self.conf.split_seed)
            split_path.parent.mkdir(parents=True, exist_ok=True)
            # unpacking train, val, test explicitly, kwd** moans otherwise
            np.savez_compressed(split_path, train=splits["train"], val=splits["val"], test=splits["test"])

        # Standardize features by removing the mean and scaling to unit variance, for training only
        scaler = StandardScaler().fit(X.iloc[splits["train"]])

        # X split has to be scaled
        X_s : dict[str, np.typing.NDArray[np.float32]] = {k: scaler.transform(X.iloc[index]).astype(np.float32) for k, index in splits.items()}

        # y split
        y_s : dict[str, np.typing.NDArray[np.int64]] = {k : y[index] for k, index in splits.items()}

        dump(value=scaler, filename= results_dir / "scaler.joblib")
        
        feature_names : list[str] = list(X.columns)
        dump(value=feature_names, filename= results_dir / "feature_names.joblib")

        return X_s, y_s , scaler, feature_names


    def make_dataloaders(self, X_s : dict[str, np.typing.NDArray[np.float32]], y_s : dict[str, np.typing.NDArray[np.int64]]) -> tuple[DataLoader, DataLoader, DataLoader]:

        train_dataset = TensorDataset(torch.from_numpy(X_s["train"]))
        val_dataset = TensorDataset(torch.from_numpy(X_s["val"]),torch.from_numpy(y_s["val"]))
        test_dataset = TensorDataset(torch.from_numpy(X_s["test"]),torch.from_numpy(y_s["test"]))
        train_dataloader= DataLoader(dataset=train_dataset, batch_size=self.conf.batch_size, shuffle=True)
        val_dataloader= DataLoader(dataset=val_dataset, batch_size=self.conf.batch_size, shuffle=False)
        test_dataloader= DataLoader(dataset=test_dataset, batch_size=self.conf.batch_size, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader

        
if __name__ == "__main__":
    conf = Config()
    my_data = Data(conf=conf)
    
    X_s, y_s, scaler, feature_names = my_data.get_splits()

    train_dataloader, val_dataloader, test_dataloader = my_data.make_dataloaders(X_s=X_s, y_s=y_s)

    # ---------- tests ----------
    df = my_data.load_and_clean(conf.data_path)
    X_raw, y_raw = my_data.build_features(df, conf.time_mode)
    with np.load(Path(conf.results_dir) / f"splits_seed_{conf.split_seed}.npz") as f:
        splits = {k: f[k] for k in f.files}

    # [1] Row counts
    print(f"[1] rows after dedup: {len(df):,} | fraud: {int(y_raw.sum())}")

    # [2] Split sizes, no fraud in train
    for k in ("train", "val", "test"):
        n_fraud = int(y_s[k].sum())
        print(f"[2] {k:5s}: {len(y_s[k]):,} rows | normal {len(y_s[k]) - n_fraud:,} | fraud {n_fraud}")
    assert y_s["train"].sum() == 0, "fraud leaked into train!"

    # [3] No overlap, full coverage
    tr, va, te = (set(splits[k].tolist()) for k in ("train", "val", "test"))
    print(f"[3] overlaps train/val {len(tr & va)}, train/test {len(tr & te)}, val/test {len(va & te)} (expect 0, 0, 0)")
    print(f"    union covers {len(tr | va | te):,} of {len(df):,} rows")

    # [4] Scaling: train exactly standardized, val/test close but not exact
    for k in ("train", "val", "test"):
        means, stds = X_s[k].mean(axis=0), X_s[k].std(axis=0)
        print(f"[4] {k:5s}: max |mean| {np.abs(means).max():.4f} | std range {stds.min():.3f} to {stds.max():.3f}")

    # [5] Signal survives scaling
    v14 = feature_names.index("V14")
    val_normal = X_s["val"][y_s["val"] == 0]
    val_fraud = X_s["val"][y_s["val"] == 1]
    print(f"[5] V14 mean: normal {val_normal[:, v14].mean():.2f} | fraud {val_fraud[:, v14].mean():.2f}")
    print(f"    mean |X|: normal {np.abs(val_normal).mean():.2f} | fraud {np.abs(val_fraud).mean():.2f}")

    # [6] Reproducibility: a fresh split with the same seed and config equals the saved file
    fresh = my_data.make_split(y_raw, conf.normal_split, conf.fraud_val_split, conf.split_seed)
    same = all(np.array_equal(fresh[k], splits[k]) for k in splits)
    print(f"[6] fresh split identical to saved file: {same} (expect True)")

    # [7] DataLoader output
    (xb,) = next(iter(train_dataloader))  # train dataset holds only X, so each batch is a 1-element list
    print(f"[7] train batch: shape {tuple(xb.shape)}, dtype {xb.dtype} (expect (256, {len(feature_names)}), torch.float32)")
    vb = next(iter(val_dataloader))
    print(f"    val batch: {len(vb)} items | X {tuple(vb[0].shape)} {vb[0].dtype} | y {tuple(vb[1].shape)} {vb[1].dtype}")

    # [8] Amount transform, before scaling
    print(f"[8] log1p(Amount): min {X_raw['Amount'].min():.2f}, max {X_raw['Amount'].max():.2f} (expect 0.00, ~10.15)")
    print(f"    {len(feature_names)} features: {feature_names}")
