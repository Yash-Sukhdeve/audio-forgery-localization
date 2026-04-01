"""FARA loss functions.

Reference: Luo et al., "A Robust Region-Aware Framework for Audio Forgery
Localization", IEEE/ACM TASLP 2026, Section III-C.
"""
from fara.losses.group_contrastive import GroupContrastiveLoss
from fara.losses.combined_loss import CombinedLoss

__all__ = ["GroupContrastiveLoss", "CombinedLoss"]
