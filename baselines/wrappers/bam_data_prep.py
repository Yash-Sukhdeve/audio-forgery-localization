"""Prepare PartialSpoof data in BAM's expected directory format.

Creates symlinks and preprocesses boundary labels. Does NOT modify
the BAM repository or the original dataset.

BAM expects:
    {data_dir}/raw/{split}/     -> audio .wav files
    {data_dir}/{split}_seglab_{resolution}.npy  -> segment labels
    {data_dir}/boundary/{utt_id}_boundary.npy  -> boundary labels per utterance

Usage:
    python baselines/wrappers/bam_data_prep.py \
        --ps_root /media/lab2208/ssd/datasets/PartialSpoof/database \
        --output_dir baselines/repos/BAM/data \
        --resolution 0.02
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np


def create_audio_symlinks(ps_root: Path, output_dir: Path):
    """Create symlinks from BAM's expected audio dirs to PartialSpoof audio."""
    for split in ["train", "dev", "eval"]:
        src = ps_root / split / "con_wav"
        dst = output_dir / "raw" / split
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if dst.is_symlink():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        os.symlink(str(src.resolve()), str(dst))
        print(f"Symlink: {dst} -> {src}")


def copy_segment_labels(ps_root: Path, output_dir: Path, resolution: float):
    """Copy segment label .npy files to BAM's expected location."""
    for split in ["train", "dev", "eval"]:
        src = ps_root / "segment_labels" / f"{split}_seglab_{resolution:.2f}.npy"
        dst = output_dir / f"{split}_seglab_{resolution:.2f}.npy"
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
            print(f"Copied: {dst}")
        else:
            print(f"Exists: {dst}")


def generate_boundary_labels(ps_root: Path, output_dir: Path, resolution: float):
    """Generate per-utterance boundary labels from segment labels.

    Boundary = 1 where segment label changes between adjacent frames.
    This replicates BAM's ps_preprocess.py logic without modifying the repo.

    NOTE: Uses RAW label convention from npy files (1=bonafide, 0=spoof).
    Boundary detection works on transitions regardless of convention.
    """
    boundary_dir = output_dir / "boundary"
    boundary_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "eval"]:
        label_path = ps_root / "segment_labels" / f"{split}_seglab_{resolution:.2f}.npy"
        seg_labels = np.load(str(label_path), allow_pickle=True).item()

        count = 0
        for utt_id, labels in seg_labels.items():
            int_labels = np.array([int(x) for x in labels], dtype=np.int64)
            # Boundary = where label changes
            boundary = np.zeros_like(int_labels)
            if len(int_labels) > 1:
                changes = np.where(int_labels[1:] != int_labels[:-1])[0]
                for idx in changes:
                    boundary[idx] = 1
                    if idx + 1 < len(boundary):
                        boundary[idx + 1] = 1

            np.save(str(boundary_dir / f"{utt_id}_boundary.npy"), boundary)
            count += 1

        print(f"Generated {count} boundary labels for {split}")


def main():
    parser = argparse.ArgumentParser(description="Prepare PartialSpoof data for BAM")
    parser.add_argument("--ps_root", type=str, required=True,
                        help="Path to PartialSpoof database/ directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for BAM data")
    parser.add_argument("--resolution", type=float, default=0.02,
                        help="Label resolution in seconds (default: 0.02 = 20ms)")
    args = parser.parse_args()

    ps_root = Path(args.ps_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_audio_symlinks(ps_root, output_dir)
    copy_segment_labels(ps_root, output_dir, args.resolution)
    generate_boundary_labels(ps_root, output_dir, args.resolution)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
