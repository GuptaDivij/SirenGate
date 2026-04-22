from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import get_window, resample_poly, stft
import torch
from torch.utils.data import Dataset

from .utils import CLASS_TO_IDX, IDX_TO_CLASS


@dataclass
class ClipRecord:
    path: str
    label_idx: int
    label_name: str
    fold: int
    clip_id: str


def is_audio_readable(path: str | Path) -> bool:
    try:
        wavfile.read(path)
        return True
    except Exception:
        return False


class UrbanSoundDataset(Dataset):
    def __init__(
        self,
        records: List[ClipRecord],
        sample_rate: int = 16000,
        clip_seconds: float = 4.0,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_seconds)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _load_audio(self, path: str) -> np.ndarray:
        sample_rate, y = wavfile.read(path)
        y = np.asarray(y)

        if y.ndim == 2:
            y = y.mean(axis=1)

        if np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.float32) / max(np.iinfo(y.dtype).max, 1)
        else:
            y = y.astype(np.float32)

        if sample_rate != self.sample_rate:
            y = resample_poly(y, self.sample_rate, sample_rate).astype(np.float32)

        if len(y) < self.clip_samples:
            y = np.pad(y, (0, self.clip_samples - len(y)))
        elif len(y) > self.clip_samples:
            start = 0
            if self.augment:
                max_offset = max(0, len(y) - self.clip_samples)
                start = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
            y = y[start : start + self.clip_samples]

        if self.augment:
            y = self._augment(y)

        return y.astype(np.float32)

    def _augment(self, y: np.ndarray) -> np.ndarray:
        noise_scale = np.random.uniform(0.0, 0.01)
        y = y + np.random.normal(0.0, noise_scale, size=y.shape).astype(np.float32)

        gain = np.random.uniform(0.9, 1.1)
        y = y * gain

        return np.clip(y, -1.0, 1.0)

    def _log_mel(self, y: np.ndarray) -> np.ndarray:
        _, _, zxx = stft(
            y,
            fs=self.sample_rate,
            window=get_window("hann", self.n_fft, fftbins=True),
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary=None,
            padded=False,
        )
        spec = np.abs(zxx) ** 2

        if spec.shape[0] < self.n_mels:
            spec = np.pad(spec, ((0, self.n_mels - spec.shape[0]), (0, 0)))

        mel_like = spec[: self.n_mels]
        mel_like = np.log(mel_like + 1e-6)
        mel_like = (mel_like - mel_like.mean()) / (mel_like.std() + 1e-6)
        return mel_like.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        y = self._load_audio(record.path)
        mel = self._log_mel(y)

        return {
            "x": torch.tensor(mel).unsqueeze(0),
            "y": torch.tensor(record.label_idx, dtype=torch.long),
            "label_name": record.label_name,
            "audio_num_samples": len(y),
            "clip_id": record.clip_id,
        }


def load_urbansound_records(dataset_root: str | Path, skip_unreadable: bool = True) -> List[ClipRecord]:
    dataset_root = Path(dataset_root)
    metadata_path = dataset_root / "metadata" / "UrbanSound8K.csv"
    return load_records_from_metadata(dataset_root, metadata_path, skip_unreadable=skip_unreadable)


def load_records_from_metadata(
    dataset_root: str | Path,
    metadata_path: str | Path,
    skip_unreadable: bool = True,
) -> List[ClipRecord]:
    dataset_root = Path(dataset_root)
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata file: {metadata_path}")

    df = pd.read_csv(metadata_path)
    records: List[ClipRecord] = []

    for _, row in df.iterrows():
        label_name = row["class"]
        if label_name not in CLASS_TO_IDX:
            continue

        fold = int(row["fold"])
        fname = row["slice_file_name"]
        path = dataset_root / "audio" / f"fold{fold}" / fname
        if skip_unreadable and not is_audio_readable(path):
            continue

        records.append(
            ClipRecord(
                path=str(path),
                label_idx=CLASS_TO_IDX[label_name],
                label_name=label_name,
                fold=fold,
                clip_id=fname,
            )
        )

    return records


def load_records_from_manifest(manifest_path: str | Path, skip_unreadable: bool = True) -> List[ClipRecord]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find manifest file: {manifest_path}")

    df = pd.read_csv(manifest_path)
    required_cols = {"path", "label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_cols)}")

    records: List[ClipRecord] = []
    for idx, row in df.iterrows():
        label_name = str(row["label"])
        if label_name not in CLASS_TO_IDX:
            continue

        clip_path = Path(str(row["path"])).expanduser()
        if not clip_path.is_absolute():
            clip_path = (manifest_path.parent / clip_path).resolve()
        if skip_unreadable and not is_audio_readable(clip_path):
            continue

        fold = int(row["fold"]) if "fold" in df.columns and pd.notna(row["fold"]) else (idx % 10) + 1
        clip_id = str(row["clip_id"]) if "clip_id" in df.columns and pd.notna(row["clip_id"]) else clip_path.name

        records.append(
            ClipRecord(
                path=str(clip_path),
                label_idx=CLASS_TO_IDX[label_name],
                label_name=label_name,
                fold=fold,
                clip_id=clip_id,
            )
        )

    return records


def split_records_by_fold(records: List[ClipRecord], val_fold: int) -> Tuple[List[ClipRecord], List[ClipRecord]]:
    train_records = [r for r in records if r.fold != val_fold]
    val_records = [r for r in records if r.fold == val_fold]
    return train_records, val_records


class SyntheticStreamingDataset(Dataset):
    """
    Lets the whole project run without UrbanSound8K.
    Creates class-specific synthetic spectrogram-like patterns.
    """

    def __init__(self, num_samples: int = 500, num_classes: int = 10, time_bins: int = 251, n_mels: int = 64):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.time_bins = time_bins
        self.n_mels = n_mels
        self.prototype_bank = self._build_prototypes()

    def _build_prototypes(self) -> np.ndarray:
        rng = np.random.default_rng(537)
        protos = []

        base_t = np.linspace(0, 2 * np.pi, self.time_bins)
        base_f = np.linspace(0, 1, self.n_mels)[:, None]

        for class_idx in range(self.num_classes):
            harmonic = np.sin(base_t * (1 + class_idx * 0.25))[None, :]
            envelope = np.cos(base_f * np.pi * (class_idx + 1))
            proto = envelope * harmonic
            proto += rng.normal(0, 0.08, size=(self.n_mels, self.time_bins))
            protos.append(proto.astype(np.float32))

        return np.stack(protos)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        label = idx % self.num_classes
        x = self.prototype_bank[label] + np.random.normal(0, 0.22, size=self.prototype_bank[label].shape)

        return {
            "x": torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            "y": torch.tensor(label, dtype=torch.long),
            "label_name": IDX_TO_CLASS[label],
            "audio_num_samples": 16000 * 4,
            "clip_id": f"synthetic_{idx}",
        }
