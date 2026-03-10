"""Prepare PartialSpoof data in BAM's expected directory format.

Creates symlinks and preprocesses boundary labels. Does NOT modify
the BAM repository or the original dataset.

BAM expects (from its ps_preprocess.py and dataset/partialspoof.py):
    {data_dir}/raw/{split}/                                    -> audio .wav files
    {data_dir}/{split}_seglab_{resolution}.npy                 -> segment labels
    {data_dir}/boundary_{resolution}_labels/{split}/{utt_id}_boundary.npy  -> boundary labels

BAM's native resolution is 0.16s (160ms). PartialSpoof provides labels at
multiple resolutions including 0.16s.

Usage:
    python baselines/wrappers/bam_data_prep.py \
        --ps_root /media/lab2208/ssd/datasets/PartialSpoof/database \
        --output_dir baselines/repos/BAM/data \
        --resolution 0.16
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
    """Copy segment label .npy files to BAM's expected location.

    BAM formats resolution as Python's default float str, e.g. 0.16 -> '0.16'.
    """
    for split in ["train", "dev", "eval"]:
        src = ps_root / "segment_labels" / f"{split}_seglab_{resolution}.npy"
        dst = output_dir / f"{split}_seglab_{resolution}.npy"
        if dst.exists():
            dst.unlink()
        shutil.copy2(str(src), str(dst))
        print(f"Copied: {dst}")


def generate_boundary_labels(ps_root: Path, output_dir: Path, resolution: float):
    """Generate per-utterance boundary labels from segment labels.

    Replicates BAM's ps_preprocess.py:get_boundary_labels() exactly:
    - Iterates over labels, tracks 'last' label
    - On transition: splice_index = i if label==0 else i-1
    - Collects unique splice indices, marks boundary[pos] = 1

    NOTE: Uses RAW label convention from npy files (1=bonafide, 0=spoof).
    BAM's boundary path format: boundary_{resolution}_labels/{split}/{utt_id}_boundary.npy
    """
    for split in ["train", "dev", "eval"]:
        boundary_dir = output_dir / f"boundary_{resolution}_labels" / split
        boundary_dir.mkdir(parents=True, exist_ok=True)

        label_path = ps_root / "segment_labels" / f"{split}_seglab_{resolution}.npy"
        seg_labels = np.load(str(label_path), allow_pickle=True).item()

        all_count = 0
        boundary_count = 0
        for utt_id, labels in seg_labels.items():
            int_labels = labels.astype(np.int32)
            all_count += len(int_labels)

            # Exact BAM logic from ps_preprocess.py
            pos = []
            last = None
            for i, label in enumerate(int_labels):
                if i == 0:
                    last = label
                if label != last:
                    splice_index = i if label == 0 else i - 1
                    pos.append(splice_index)
                    last = label

            pos = list(set(pos))
            boundary_count += len(pos)
            boundary_label = np.zeros_like(int_labels)
            if pos:
                boundary_label[pos] = 1.0
            np.save(str(boundary_dir / f"{utt_id}_boundary.npy"), boundary_label)

        ratio = (all_count - boundary_count) / max(boundary_count, 1)
        print(f"[{split}] Generated {len(seg_labels)} boundary labels, "
              f"pos_weight: {ratio:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Prepare PartialSpoof data for BAM")
    parser.add_argument("--ps_root", type=str, required=True,
                        help="Path to PartialSpoof database/ directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for BAM data")
    parser.add_argument("--resolution", type=float, default=0.16,
                        help="Label resolution in seconds (default: 0.16 = 160ms, BAM native)")
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
