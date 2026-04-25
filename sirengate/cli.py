from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from .calibration import TemperatureScaler
from .config import Config
from .data import (
    SyntheticStreamingDataset,
    UrbanSoundDataset,
    load_records_from_manifest,
    load_urbansound_records,
    split_records_by_fold,
    summarize_urbansound_dataset,
)
from .models import SmallAudioCNN, collect_logits, train_model
from .plots import plot_threshold_trace, plot_tradeoff
from .simulation import evaluate_policies_from_logits, sweep_thresholds_from_logits
from .utils import ensure_dir, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SirenGate CLI")
    parser.add_argument("command", choices=["demo", "describe", "train", "simulate", "sweep"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default="results/run")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    return parser


def load_records(dataset_root: str | None = None, manifest: str | None = None):
    if manifest:
        return load_records_from_manifest(manifest)
    if dataset_root:
        return load_urbansound_records(dataset_root)
    raise SystemExit("Either --dataset-root or --manifest is required")


def build_model(config: Config) -> SmallAudioCNN:
    return SmallAudioCNN(num_classes=10, embedding_dim=config.embedding_dim)


def run_demo(config: Config, output_dir: Path) -> None:
    dataset = SyntheticStreamingDataset(num_samples=600, n_mels=config.n_mels)

    train_ds, val_ds = random_split(
        dataset,
        [400, 200],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cpu")
    model = build_model(config)

    artifacts = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=min(config.epochs, 5),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        output_dir=output_dir,
        device=device,
    )

    model.load_state_dict(torch.load(artifacts.best_model_path, map_location=device))
    logits, labels, embeddings = collect_logits(model, val_loader, device)
    scaler = TemperatureScaler().fit(logits, labels)

    results = evaluate_policies_from_logits(
        logits=logits,
        labels=labels,
        embeddings=embeddings,
        clip_seconds=config.clip_seconds,
        scaler=scaler,
        config=config,
        output_dir=output_dir,
    )

    plot_tradeoff(output_dir / "policy_summary.csv", output_dir / "bandwidth_vs_f1.png")
    plot_threshold_trace(output_dir / "events_adaptive_triage.csv", output_dir / "adaptive_threshold_trace.png")

    save_json(
        {
            "mode": "demo",
            "best_model_path": artifacts.best_model_path,
            "val_accuracy": artifacts.val_accuracy,
            "temperature": scaler.temperature,
            "policies": {k: v.summary for k, v in results.items()},
        },
        output_dir / "run_summary.json",
    )


def run_describe(dataset_root: str, output_dir: Path) -> None:
    summary = summarize_urbansound_dataset(dataset_root)
    save_json(summary, output_dir / "dataset_summary.json")


def run_train(config: Config, dataset_root: str | None, manifest: str | None, output_dir: Path) -> None:
    records = load_records(dataset_root, manifest)
    train_records, val_records = split_records_by_fold(records, config.val_fold)

    train_ds = UrbanSoundDataset(
        train_records,
        sample_rate=config.sample_rate,
        clip_seconds=config.clip_seconds,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        augment=True,
    )
    val_ds = UrbanSoundDataset(
        val_records,
        sample_rate=config.sample_rate,
        clip_seconds=config.clip_seconds,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    device = torch.device("cpu")
    model = build_model(config)

    artifacts = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        output_dir=output_dir,
        device=device,
    )

    model.load_state_dict(torch.load(artifacts.best_model_path, map_location=device))
    logits, labels, embeddings = collect_logits(model, val_loader, device)
    scaler = TemperatureScaler().fit(logits, labels)

    pd.DataFrame(artifacts.history).to_csv(output_dir / "train_history.csv", index=False)

    save_json(
        {
            "best_model_path": artifacts.best_model_path,
            "best_val_accuracy": artifacts.val_accuracy,
            "temperature": scaler.temperature,
            "dataset_root": dataset_root,
            "manifest": manifest,
            "num_train_clips": len(train_records),
            "num_val_clips": len(val_records),
            "embedding_dim": config.embedding_dim,
            "mean_embedding_l2": float((torch.tensor(embeddings).norm(dim=1).mean()).item()),
        },
        output_dir / "train_summary.json",
    )


def run_simulate(config: Config, dataset_root: str | None, manifest: str | None, checkpoint: str, output_dir: Path) -> None:
    records = load_records(dataset_root, manifest)
    _, val_records = split_records_by_fold(records, config.val_fold)

    val_ds = UrbanSoundDataset(
        val_records,
        sample_rate=config.sample_rate,
        clip_seconds=config.clip_seconds,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        augment=False,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    device = torch.device("cpu")
    model = build_model(config)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    logits, labels, embeddings = collect_logits(model, val_loader, device)
    scaler = TemperatureScaler().fit(logits, labels)

    results = evaluate_policies_from_logits(
        logits=logits,
        labels=labels,
        embeddings=embeddings,
        clip_seconds=config.clip_seconds,
        scaler=scaler,
        config=config,
        output_dir=output_dir,
    )

    plot_tradeoff(output_dir / "policy_summary.csv", output_dir / "bandwidth_vs_f1.png")
    plot_threshold_trace(output_dir / "events_adaptive_triage.csv", output_dir / "adaptive_threshold_trace.png")

    save_json(
        {
            "checkpoint": checkpoint,
            "temperature": scaler.temperature,
            "dataset_root": dataset_root,
            "manifest": manifest,
            "policies": {k: v.summary for k, v in results.items()},
        },
        output_dir / "simulation_summary.json",
    )


def run_sweep(
    config: Config,
    dataset_root: str | None,
    manifest: str | None,
    checkpoint: str,
    thresholds: list[float] | None,
    output_dir: Path,
) -> None:
    records = load_records(dataset_root, manifest)
    _, val_records = split_records_by_fold(records, config.val_fold)

    val_ds = UrbanSoundDataset(
        val_records,
        sample_rate=config.sample_rate,
        clip_seconds=config.clip_seconds,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        augment=False,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    device = torch.device("cpu")
    model = build_model(config)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    logits, labels, _ = collect_logits(model, val_loader, device)
    scaler = TemperatureScaler().fit(logits, labels)

    sweep_thresholds = thresholds or config.threshold_sweep
    if not sweep_thresholds:
        raise SystemExit("Provide --thresholds or set threshold_sweep in the config")

    sweep_df = sweep_thresholds_from_logits(
        logits=logits,
        labels=labels,
        clip_seconds=config.clip_seconds,
        scaler=scaler,
        config=config,
        thresholds=list(sweep_thresholds),
        output_csv=output_dir / "threshold_sweep.csv",
    )
    plot_tradeoff(output_dir / "threshold_sweep.csv", output_dir / "threshold_sweep_tradeoff.png")
    save_json(
        {
            "checkpoint": checkpoint,
            "thresholds": list(map(float, sweep_thresholds)),
            "best_macro_f1_threshold": float(sweep_df.sort_values("macro_f1", ascending=False).iloc[0]["threshold"]),
        },
        output_dir / "threshold_sweep_summary.json",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    output_dir = ensure_dir(args.output_dir)
    set_seed(config.seed)

    if args.command == "demo":
        run_demo(config, output_dir)
    elif args.command == "describe":
        if not args.dataset_root:
            raise SystemExit("--dataset-root is required for describe")
        run_describe(args.dataset_root, output_dir)
    elif args.command == "train":
        if not args.dataset_root and not args.manifest:
            raise SystemExit("--dataset-root or --manifest is required for train")
        run_train(config, args.dataset_root, args.manifest, output_dir)
    elif args.command == "simulate":
        if (not args.dataset_root and not args.manifest) or not args.checkpoint:
            raise SystemExit("--checkpoint and either --dataset-root or --manifest are required for simulate")
        run_simulate(config, args.dataset_root, args.manifest, args.checkpoint, output_dir)
    elif args.command == "sweep":
        if (not args.dataset_root and not args.manifest) or not args.checkpoint:
            raise SystemExit("--checkpoint and either --dataset-root or --manifest are required for sweep")
        run_sweep(config, args.dataset_root, args.manifest, args.checkpoint, args.thresholds, output_dir)


if __name__ == "__main__":
    main()
