import argparse
import torch
from dataclasses import fields, replace, asdict
from pathlib import Path
from config import Config
from vae import ACTIVATION_MAP
from score import SCORERS
from evaluate import THRESHOLDS

# names of all Config fields, used to pick the CLI arguments that override the config
CONFIG_FIELDS: set[str] = {f.name for f in fields(Config)}


def build_parser() -> argparse.ArgumentParser:

    defaults = Config()
    parser = argparse.ArgumentParser(description="VAE-based credit card fraud detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # options every command needs; add_help=False because it is only used as a parent
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-name", help=f"folder inside results/ to write to / read from (default: {defaults.run_name})")
    common.add_argument("--seed", type=int, help=f"seed for training and stochastic scoring (default: {defaults.seed})")
    common.add_argument("--score-method", choices=list(SCORERS.keys()), help=f"anomaly score (default: {defaults.score_method})")

    train = subparsers.add_parser("train", parents=[common], help="train a VAE on normal transactions")
    train.add_argument("--data-path", help=f"default: {defaults.data_path}")
    train.add_argument("--time-mode", choices=["drop", "cyclic"], help=f"default: {defaults.time_mode}")
    train.add_argument("--latent-dim", type=int, help=f"default: {defaults.latent_dim}")
    train.add_argument("--hidden-dims", type=int, nargs="+", help=f"encoder layer sizes, decoder uses them reversed (default: {' '.join(map(str, defaults.hidden_dims))})")
    train.add_argument("--activation", choices=list(ACTIVATION_MAP.keys()), help=f"default: {defaults.activation.__name__}")
    train.add_argument("--beta", type=float, help=f"weight of the KL term (default: {defaults.beta})")
    train.add_argument("--lr", type=float, help=f"default: {defaults.lr}")
    train.add_argument("--batch-size", type=int, help=f"default: {defaults.batch_size}")
    train.add_argument("--max-epochs", type=int, help=f"default: {defaults.max_epochs}")
    train.add_argument("--patience", type=int, help=f"early stopping patience (default: {defaults.patience})")
    train.add_argument("--kl-warmup-epochs", type=int, help=f"default: {defaults.kl_warmup_epochs}")
    train.add_argument("--grad-clip", type=float, help=f"0 disables clipping (default: {defaults.grad_clip})")
    train.add_argument("--select-by", choices=["val_ap", "val_loss"], help=f"default: {defaults.select_by}")
    train.add_argument("--overfit-check", action="store_true", help="only run the 512-sample overfit sanity check")

    threshold = subparsers.add_parser("threshold", parents=[common], help="choose a decision threshold on the validation set")
    threshold.add_argument("--threshold-method", choices=list(THRESHOLDS.keys()), help=f"default: {defaults.threshold_method}")
    threshold.add_argument("--threshold-percentile", type=float, help=f"for 'percentile' (default: {defaults.threshold_percentile})")
    threshold.add_argument("--recall-target", type=float, help=f"for 'recall_target' (default: {defaults.recall_target})")

    evaluate = subparsers.add_parser("evaluate", parents=[common], help="compute metrics and figures")
    evaluate.add_argument("--split", choices=["val", "test"], default="val")
    evaluate.add_argument("--force", action="store_true", help="allow overwriting test_metrics.json")

    generate = subparsers.add_parser("generate", parents=[common], help="sample synthetic transactions from the prior")
    generate.add_argument("--n-samples", type=int, default=5000)

    predict = subparsers.add_parser("predict", parents=[common], help="score transactions from a CSV file")
    predict.add_argument("--input", required=True, help="CSV with columns V1..V28 and Amount (Class optional)")
    predict.add_argument("--output", help="default: results/<run-name>/predictions.csv")

    return parser


def config_from_checkpoint(run_name: str, results_dir: str) -> Config:

    # rebuild the Config a run was trained with, so evaluation uses the same features and architecture
    checkpoint = torch.load(Path(results_dir) / run_name / "model.pt", weights_only=False)
    saved: dict = dict(checkpoint["config"])
    saved["activation"] = ACTIVATION_MAP[saved["activation"]]
    # Config uses tuples for these, make sure the rebuilt config does too
    for key in ("hidden_dims", "normal_split"):
        saved[key] = tuple(saved[key])
    return Config(**{k: v for k, v in saved.items() if k in CONFIG_FIELDS})


def print_config(conf: Config, command: str) -> None:

    print(f"\n{'=' * 20} {command.upper()} | run '{conf.run_name}' {'=' * 20}")
    for key, value in asdict(conf).items():
        if isinstance(value, type):
            value = value.__name__
        print(f"  {key:22s} {value}")
    print()


def main() -> None:

    args = build_parser().parse_args()

    # only arguments the user actually typed are not None, those override the config
    overrides: dict = {k: v for k, v in vars(args).items() if k in CONFIG_FIELDS and v is not None}
    if "activation" in overrides:
        overrides["activation"] = ACTIVATION_MAP[overrides["activation"]]
    if "hidden_dims" in overrides:
        overrides["hidden_dims"] = tuple(overrides["hidden_dims"])

    if args.command == "train":
        conf = replace(Config(), **overrides)
    else:
        run_name = overrides.get("run_name", Config.run_name)
        conf = replace(config_from_checkpoint(run_name, Config.results_dir), **overrides)

    print_config(conf, args.command)

    # imported only when needed, so each command loads just the module it uses
    if args.command == "train":
        from train import fit, overfit_check
        if args.overfit_check:
            overfit_check(conf)
        else:
            fit(conf)
    elif args.command == "threshold":
        from evaluate import choose_threshold
        choose_threshold(conf)
    elif args.command == "evaluate":
        from evaluate import run_evaluation
        run_evaluation(conf, split=args.split, force=args.force)
    elif args.command == "generate":
        from generate import generate
        generate(conf, n_samples=args.n_samples)
    elif args.command == "predict":
        from predict import predict
        predict(conf, input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
