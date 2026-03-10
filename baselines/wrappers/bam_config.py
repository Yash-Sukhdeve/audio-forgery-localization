"""BAM configuration overrides.

Generates a modified config that points to our WavLM checkpoint
without modifying the BAM repository.
"""
import yaml
from pathlib import Path


def create_bam_config(
    bam_repo_dir: str,
    wavlm_ckpt: str,
    output_path: str,
) -> str:
    """Create BAM config with our WavLM checkpoint path.

    Reads BAM's default config and overrides ssl_ckpt.
    Writes to output_path (outside repo). Returns output path.
    """
    bam_repo = Path(bam_repo_dir)
    default_config = bam_repo / "config" / "bam_wavlm.yaml"

    with open(default_config) as f:
        config = yaml.safe_load(f)

    config["ssl_ckpt"] = str(Path(wavlm_ckpt).resolve())

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.dump(config, f)

    print(f"Config written to {output}")
    return str(output)
