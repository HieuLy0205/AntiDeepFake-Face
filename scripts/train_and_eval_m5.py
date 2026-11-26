"""Train (Frozen Linear Probe + Top-2-Blocks Fine-Tuning) and run final E5 test evaluation (M5).

Methodology (zero data leakage):
- Uses the strict 3-way subject-disjoint split in `data/rtfs_m5_rich_metadata.csv`:
  train (480 videos, 350 subjects) / val (120 videos, 100 subjects) / test (200 videos, 164 subjects).
- Extracts frozen stem hidden states (after encoder layer 9 of 12) once for all splits
  (with 2 views on train: clean + forensic-augmented via hflip, color jitter, JPEG compression).
- Compares on identical held-out `test` split (thresholds calibrated strictly on `val`):
  1. Zero-shot Pretrained baselines (`king1oo1/deepfake-model`, `Deep-Fake-Detector-v2-Model`, 50/50 Ensemble)
  2. Frozen-backbone L2-normalized Linear Probes (UniversalFakeDetect / GenD-linear ablation)
  3. Top-2 Transformer Blocks + Final LayerNorm + Classifier fine-tuning (exact full-model parity,
     zero extra inference latency on CPU) for both SigLIP2-Base and ViT-Base, plus their 50/50 Ensemble.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import io
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deepfake.predictor import DeepfakeDetector, resolve_fake_label_index  # noqa: E402
from src.schemas import FaceCropResult  # noqa: E402


BACKBONES = [
    ("siglip2", "king1oo1/deepfake-model"),
    ("vit", "prithivMLmods/Deep-Fake-Detector-v2-Model"),
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def forensic_augment(arr_rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply realistic video-call forensic augmentations (flip, lighting, JPEG re-encoding)."""
    img = Image.fromarray(arr_rgb, mode="RGB")
    if rng.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    # Mild crop & resize (scale 0.88 - 1.0)
    if rng.random() < 0.6:
        w, h = img.size
        scale = rng.uniform(0.88, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        x0 = rng.randint(0, max(0, w - nw))
        y0 = rng.randint(0, max(0, h - nh))
        img = img.crop((x0, y0, x0 + nw, y0 + nh)).resize((w, h), Image.Resampling.BILINEAR)
    # Mild brightness/contrast jitter
    if rng.random() < 0.6:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    # JPEG compression simulation (video codec artifact robustness)
    if rng.random() < 0.7:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(60, 95))
        buf.seek(0)
        with Image.open(buf) as dec:
            img = dec.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_cached_crops(
    rows: List[dict],
    cache_dir: Path,
) -> Dict[str, List[np.ndarray]]:
    crops: Dict[str, List[np.ndarray]] = {}
    for r in rows:
        sid = r["sample_id"]
        sdir = cache_dir / sid
        imgs = []
        for p in sorted(sdir.glob("*.png")):
            with Image.open(p) as im:
                imgs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
        if not imgs:
            raise FileNotFoundError(f"No cached PNG crops found in {sdir}")
        crops[sid] = imgs
    return crops


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(float(roc_auc_score(yt, y_score[idx])))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if aucs else (0.0, 0.0)


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    acers = 0.5 * (fpr + (1.0 - tpr))
    best_idx = int(np.argmin(acers))
    eer_idx = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    eer = float(0.5 * (fpr[eer_idx] + (1.0 - tpr[eer_idx])))
    thr = float(thresholds[best_idx])
    return (thr if np.isfinite(thr) else 0.5), eer


def compute_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    apcer = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    bpcer = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    acer = 0.5 * (apcer + bpcer)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_fake": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "apcer": round(apcer, 4),
        "bpcer": round(bpcer, 4),
        "acer": round(acer, 4),
        "confusion_matrix": {
            "tn_real": int(tn),
            "fp_real_as_fake": int(fp),
            "fn_fake_as_real": int(fn),
            "tp_fake": int(tp),
        },
    }


def evaluate_video_scores(
    rows: List[dict],
    video_scores: Dict[str, float],
    threshold: float,
    compute_ci: bool = True,
) -> Dict[str, Any]:
    y_true = np.asarray([1 if r["label"] == "Deepfake" else 0 for r in rows], dtype=int)
    y_score = np.asarray([video_scores[r["sample_id"]] for r in rows], dtype=float)

    auc = float(roc_auc_score(y_true, y_score))
    ci = bootstrap_auc_ci(y_true, y_score) if compute_ci else (auc, auc)
    pr_auc = float(average_precision_score(y_true, y_score))
    opt_thr, eer = find_optimal_threshold(y_true, y_score)

    def _slice(mask_fn) -> Dict[str, Any]:
        sub_rows = [r for r in rows if mask_fn(r)]
        yt = np.asarray([1 if r["label"] == "Deepfake" else 0 for r in sub_rows], dtype=int)
        ys = np.asarray([video_scores[r["sample_id"]] for r in sub_rows], dtype=float)
        out: Dict[str, Any] = {
            "n_samples": len(sub_rows),
            "n_real": int(np.sum(yt == 0)),
            "n_fake": int(np.sum(yt == 1)),
            "mean_prob_real": round(float(np.mean(ys[yt == 0])), 4) if np.any(yt == 0) else None,
            "mean_prob_fake": round(float(np.mean(ys[yt == 1])), 4) if np.any(yt == 1) else None,
        }
        if len(np.unique(yt)) == 2:
            out["roc_auc"] = round(float(roc_auc_score(yt, ys)), 4)
            m = compute_threshold_metrics(yt, ys, threshold)
            out["apcer"] = m["apcer"]
            out["bpcer"] = m["bpcer"]
            out["acer"] = m["acer"]
            out["recall_fake"] = round(1.0 - m["apcer"], 4)
        return out

    slices = {
        "real_vs_inswapper_all": _slice(lambda r: r["generator"] in ("original", "inswapper")),
        "real_vs_inswapper_no_enhancer": _slice(
            lambda r: r["generator"] == "original" or (r["generator"] == "inswapper" and r["has_faceenhancer"] == "False")
        ),
        "real_vs_inswapper_with_enhancer": _slice(
            lambda r: r["generator"] == "original" or (r["generator"] == "inswapper" and r["has_faceenhancer"] == "True")
        ),
        "real_vs_uniface_all": _slice(lambda r: r["generator"] in ("original", "uniface")),
        "real_vs_uniface_no_enhancer": _slice(
            lambda r: r["generator"] == "original" or (r["generator"] == "uniface" and r["has_faceenhancer"] == "False")
        ),
        "real_vs_uniface_with_enhancer": _slice(
            lambda r: r["generator"] == "original" or (r["generator"] == "uniface" and r["has_faceenhancer"] == "True")
        ),
        "all_fake_no_enhancer": _slice(lambda r: r["generator"] == "original" or r["has_faceenhancer"] == "False"),
        "all_fake_with_enhancer": _slice(lambda r: r["generator"] == "original" or r["has_faceenhancer"] == "True"),
        "lighting_bright": _slice(lambda r: r["lighting"] == "bright"),
        "lighting_average": _slice(lambda r: r["lighting"] == "average"),
        "lighting_dark": _slice(lambda r: r["lighting"] == "dark"),
        "gender_man": _slice(lambda r: r["gender"] == "man"),
        "gender_woman": _slice(lambda r: r["gender"] == "woman"),
    }

    return {
        "n_videos": len(rows),
        "n_real": int(np.sum(y_true == 0)),
        "n_fake": int(np.sum(y_true == 1)),
        "roc_auc": round(auc, 4),
        "roc_auc_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "pr_auc": round(pr_auc, 4),
        "eer": round(eer, 4),
        "self_optimal_threshold": round(opt_thr, 4),
        "at_val_tau": compute_threshold_metrics(y_true, y_score, threshold),
        "at_0_50": compute_threshold_metrics(y_true, y_score, 0.50),
        "slices": slices,
    }


def aggregate_frame_probs(frame_sids: Sequence[str], frame_probs: Sequence[float]) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for sid, p in zip(frame_sids, frame_probs):
        buckets.setdefault(sid, []).append(float(p))
    return {sid: float(np.mean(vals)) for sid, vals in buckets.items()}


class SplitTopHead(nn.Module):
    """Trainable top-K Transformer blocks + final LayerNorm + Linear Classifier (exact HF parity)."""

    def __init__(self, hf_model: nn.Module, freeze_blocks: int = 10, arch: str = "siglip2") -> None:
        super().__init__()
        self.arch = arch
        self.freeze_blocks = freeze_blocks
        if arch == "siglip2":
            vm = hf_model.vision_model
            self.layers = copy.deepcopy(vm.encoder.layers[freeze_blocks:])
            self.norm = copy.deepcopy(vm.post_layernorm)
            self.classifier = copy.deepcopy(hf_model.classifier)
        elif arch == "vit":
            vit = hf_model.vit
            self.layers = copy.deepcopy(vit.layers[freeze_blocks:])
            self.norm = copy.deepcopy(vit.layernorm)
            self.classifier = copy.deepcopy(hf_model.classifier)
        else:
            raise ValueError(f"Unsupported arch: {arch}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_dropout: float = 0.0,
        noise_std: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = hidden_states.float()
        if self.training:
            if noise_std > 0:
                h = h + torch.randn_like(h) * noise_std
            if token_dropout > 0:
                h = F.dropout(h, p=token_dropout, training=True)

        if self.arch == "siglip2":
            for layer in self.layers:
                out = layer(h, attention_mask=None)
                h = out[0] if isinstance(out, tuple) else out
            h = self.norm(h)
            pooled = h.mean(dim=1)
        else:
            for layer in self.layers:
                out = layer(h)
                h = out[0] if isinstance(out, tuple) else out
            h = self.norm(h)
            pooled = h[:, 0, :]

        logits = self.classifier(pooled)
        return logits, pooled

    def copy_into_hf_model(self, hf_model: nn.Module) -> None:
        """Load trained top blocks back into the original HF model for exact standard inference."""
        if self.arch == "siglip2":
            vm = hf_model.vision_model
            vm.encoder.layers[self.freeze_blocks :].load_state_dict(self.layers.state_dict())
            vm.post_layernorm.load_state_dict(self.norm.state_dict())
            hf_model.classifier.load_state_dict(self.classifier.state_dict())
        else:
            vit = hf_model.vit
            vit.layers[self.freeze_blocks :].load_state_dict(self.layers.state_dict())
            vit.layernorm.load_state_dict(self.norm.state_dict())
            hf_model.classifier.load_state_dict(self.classifier.state_dict())

    def export_adapter_state(self, model_id: str, id2label: Dict[int, str], fake_idx: int, val_tau: float) -> Dict[str, Any]:
        """Serialize only the modified top-K block weights (FP16 for compact storage)."""
        state_fp16 = {}
        if self.arch == "siglip2":
            prefix_layers = "vision_model.encoder.layers"
            prefix_norm = "vision_model.post_layernorm"
        else:
            prefix_layers = "vit.layers"
            prefix_norm = "vit.layernorm"

        for i, layer in enumerate(self.layers):
            layer_idx = self.freeze_blocks + i
            for k, v in layer.state_dict().items():
                state_fp16[f"{prefix_layers}.{layer_idx}.{k}"] = v.detach().cpu().to(torch.float16)
        for k, v in self.norm.state_dict().items():
            state_fp16[f"{prefix_norm}.{k}"] = v.detach().cpu().to(torch.float16)
        for k, v in self.classifier.state_dict().items():
            state_fp16[f"classifier.{k}"] = v.detach().cpu().to(torch.float16)

        return {
            "format": "adf_m5_partial_adapter_v1",
            "model_id": model_id,
            "arch": self.arch,
            "freeze_blocks": self.freeze_blocks,
            "id2label": id2label,
            "fake_label_index": fake_idx,
            "recommended_threshold": round(float(val_tau), 4),
            "state_dict": state_fp16,
        }


def extract_stem_and_zeroshot(
    detector: DeepfakeDetector,
    arch: str,
    freeze_blocks: int,
    images_rgb: Sequence[np.ndarray],
    batch_size: int = 32,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Run frozen stem (layers[:freeze_blocks]) AND zero-shot tail in one pass.

    Returns:
      - stem_hidden: (N, seq_len, hidden_dim) float16 tensor after frozen stem
      - final_emb:   (N, hidden_dim) float32 L2-normalized embedding from zero-shot model
      - zs_probs:    (N,) float32 zero-shot fake probabilities
    """
    model = detector._model
    processor = detector._processor
    fake_idx = int(detector.fake_label_index)

    stem_chunks: List[torch.Tensor] = []
    emb_chunks: List[np.ndarray] = []
    prob_chunks: List[np.ndarray] = []

    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(images_rgb), batch_size):
            batch_imgs = [Image.fromarray(arr, mode="RGB") for arr in images_rgb[i : i + batch_size]]
            pv = processor(images=batch_imgs, return_tensors="pt")["pixel_values"]

            if arch == "siglip2":
                vm = model.vision_model
                h = vm.embeddings(pv)
                for layer in vm.encoder.layers[:freeze_blocks]:
                    out = layer(h, attention_mask=None)
                    h = out[0] if isinstance(out, tuple) else out
                stem_h = h
                h_tail = stem_h
                for layer in vm.encoder.layers[freeze_blocks:]:
                    out = layer(h_tail, attention_mask=None)
                    h_tail = out[0] if isinstance(out, tuple) else out
                h_tail = vm.post_layernorm(h_tail)
                pooled = h_tail.mean(dim=1)
                logits = model.classifier(pooled)
            else:
                vit = model.vit
                h = vit.embeddings(pv)
                for layer in vit.layers[:freeze_blocks]:
                    out = layer(h)
                    h = out[0] if isinstance(out, tuple) else out
                stem_h = h
                h_tail = stem_h
                for layer in vit.layers[freeze_blocks:]:
                    out = layer(h_tail)
                    h_tail = out[0] if isinstance(out, tuple) else out
                h_tail = vit.layernorm(h_tail)
                pooled = h_tail[:, 0, :]
                logits = model.classifier(pooled)

            probs = torch.softmax(logits, dim=-1)[:, fake_idx]
            normed_emb = F.normalize(pooled, p=2, dim=-1)

            stem_chunks.append(stem_h.to(torch.float16).cpu())
            emb_chunks.append(normed_emb.cpu().numpy().astype(np.float32))
            prob_chunks.append(probs.cpu().numpy().astype(np.float32))

            done = min(i + batch_size, len(images_rgb))
            if (i // batch_size) % 25 == 0 or done == len(images_rgb):
                print(f"    [Stem {arch}] {done}/{len(images_rgb)} frames ({time.perf_counter() - t0:.1f}s)")

    return (
        torch.cat(stem_chunks, dim=0),
        np.concatenate(emb_chunks, axis=0),
        np.concatenate(prob_chunks, axis=0),
    )


def fit_frozen_linear_probe(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    val_emb: np.ndarray,
    val_sids: Sequence[str],
    val_rows: List[dict],
    test_emb: np.ndarray,
    test_sids: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    """Train L2-regularized balanced Logistic Regression on frozen embeddings; tune C strictly on val."""
    val_y_vid = np.asarray([1 if r["label"] == "Deepfake" else 0 for r in val_rows], dtype=int)
    best_c = 1.0
    best_val_auc = -1.0
    best_clf = None

    for c_val in (0.05, 0.2, 1.0, 5.0, 20.0):
        clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(train_emb, train_y)
        val_p = clf.predict_proba(val_emb)[:, 1]
        val_scores = aggregate_frame_probs(val_sids, val_p)
        auc = float(roc_auc_score(val_y_vid, [val_scores[r["sample_id"]] for r in val_rows]))
        if auc > best_val_auc:
            best_val_auc = auc
            best_c = c_val
            best_clf = clf

    assert best_clf is not None
    val_scores = aggregate_frame_probs(val_sids, best_clf.predict_proba(val_emb)[:, 1])
    test_scores = aggregate_frame_probs(test_sids, best_clf.predict_proba(test_emb)[:, 1])
    val_tau, _ = find_optimal_threshold(val_y_vid, np.asarray([val_scores[r["sample_id"]] for r in val_rows]))
    return val_scores, test_scores, val_tau, best_c


def train_top_blocks(
    hf_model: nn.Module,
    arch: str,
    freeze_blocks: int,
    fake_idx: int,
    train_stem_clean: torch.Tensor,
    train_stem_aug: torch.Tensor,
    train_y_fake: np.ndarray,
    val_stem: torch.Tensor,
    val_sids: Sequence[str],
    val_rows: List[dict],
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    seed: int = 42,
) -> Tuple[SplitTopHead, List[Dict[str, Any]], Dict[str, float]]:
    """Fine-tune top-K blocks + final LayerNorm + classifier on cached stem states."""
    set_seed(seed)
    head = SplitTopHead(hf_model, freeze_blocks=freeze_blocks, arch=arch)

    # Convert binary label (1=Fake, 0=Real) -> model's native class index
    real_idx = 1 - fake_idx
    native_y = np.where(train_y_fake == 1, fake_idx, real_idx).astype(np.int64)
    y_tensor = torch.from_numpy(native_y)

    # Balanced class weights (3:1 Real:Fake to counter 25%/75% class imbalance)
    n_real = max(1, int(np.sum(train_y_fake == 0)))
    n_fake = max(1, int(np.sum(train_y_fake == 1)))
    w_real = (n_real + n_fake) / (2.0 * n_real)
    w_fake = (n_real + n_fake) / (2.0 * n_fake)
    class_weights = torch.zeros(2, dtype=torch.float32)
    class_weights[real_idx] = w_real
    class_weights[fake_idx] = w_fake

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Slightly higher LR for classifier head than transformer blocks
    param_groups = [
        {"params": list(head.layers.parameters()) + list(head.norm.parameters()), "lr": lr},
        {"params": list(head.classifier.parameters()), "lr": lr * 3.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    val_y_vid = np.asarray([1 if r["label"] == "Deepfake" else 0 for r in val_rows], dtype=int)

    def _eval_split(stem_tensor: torch.Tensor, sids: Sequence[str]) -> Dict[str, float]:
        head.eval()
        probs_all: List[float] = []
        with torch.inference_mode():
            for i in range(0, len(stem_tensor), 64):
                logits, _ = head(stem_tensor[i : i + 64])
                p = torch.softmax(logits, dim=-1)[:, fake_idx]
                probs_all.extend(p.cpu().tolist())
        return aggregate_frame_probs(sids, probs_all)

    best_val_auc = -1.0
    best_state = copy.deepcopy(head.state_dict())
    best_val_scores: Dict[str, float] = {}
    history: List[Dict[str, Any]] = []

    n_train = len(train_stem_clean)
    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        head.train()
        perm = torch.randperm(n_train)
        running_loss = 0.0
        steps = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            # Mix clean and forensic-augmented stem views 50/50 per sample
            use_aug = torch.rand(len(idx)) < 0.5
            x_batch = torch.where(
                use_aug.view(-1, 1, 1),
                train_stem_aug[idx],
                train_stem_clean[idx],
            )
            y_batch = y_tensor[idx]

            optimizer.zero_grad(set_to_none=True)
            logits, _ = head(x_batch, token_dropout=0.10, noise_std=0.03)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            steps += 1

        scheduler.step()
        val_scores = _eval_split(val_stem, val_sids)
        val_auc = float(roc_auc_score(val_y_vid, [val_scores[r["sample_id"]] for r in val_rows]))
        val_thr, val_eer = find_optimal_threshold(val_y_vid, np.asarray([val_scores[r["sample_id"]] for r in val_rows]))
        avg_loss = running_loss / max(1, steps)
        ep_sec = time.perf_counter() - t_ep

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_state = copy.deepcopy(head.state_dict())
            best_val_scores = val_scores

        history.append(
            {
                "epoch": ep,
                "train_loss": round(avg_loss, 4),
                "val_roc_auc": round(val_auc, 4),
                "val_eer": round(val_eer, 4),
                "val_opt_thr": round(val_thr, 4),
                "epoch_sec": round(ep_sec, 1),
                "is_best": is_best,
            }
        )
        mark = " *" if is_best else ""
        print(
            f"    [Train {arch} ep {ep:02d}/{epochs}] loss={avg_loss:.4f} | "
            f"val_AUC={val_auc:.4f} | val_EER={val_eer:.4f} | {ep_sec:.1f}s{mark}"
        )

    head.load_state_dict(best_state)
    head.eval()
    return head, history, best_val_scores


def score_with_head(head: SplitTopHead, stem_tensor: torch.Tensor, sids: Sequence[str], fake_idx: int) -> Dict[str, float]:
    head.eval()
    probs_all: List[float] = []
    with torch.inference_mode():
        for i in range(0, len(stem_tensor), 64):
            logits, _ = head(stem_tensor[i : i + 64])
            p = torch.softmax(logits, dim=-1)[:, fake_idx]
            probs_all.extend(p.cpu().tolist())
    return aggregate_frame_probs(sids, probs_all)


def measure_cpu_latency(detector: DeepfakeDetector, sample_face: np.ndarray) -> Dict[str, float]:
    dummy = FaceCropResult(frame_id=0, has_face=True, bbox=(0, 0, 224, 224), det_confidence=0.99, face_rgb=sample_face)
    for _ in range(3):
        detector.predict(dummy)
    b1 = []
    for _ in range(15):
        t0 = time.perf_counter()
        detector.predict(dummy)
        b1.append((time.perf_counter() - t0) * 1000.0)

    batch10 = [
        FaceCropResult(frame_id=i, has_face=True, bbox=(0, 0, 224, 224), det_confidence=0.99, face_rgb=sample_face)
        for i in range(10)
    ]
    b10 = []
    for _ in range(4):
        t0 = time.perf_counter()
        detector.predict_batch(batch10)
        b10.append((time.perf_counter() - t0) * 1000.0)

    m1 = float(np.mean(b1))
    m10 = float(np.mean(b10))
    return {
        "batch1_mean_ms": round(m1, 2),
        "batch1_fps": round(1000.0 / m1, 2),
        "batch10_total_ms": round(m10, 2),
        "batch10_ms_per_frame": round(m10 / 10.0, 2),
        "batch10_fps": round(10000.0 / m10, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate M5 Deepfake detectors on RTFS-800.")
    parser.add_argument("--metadata", type=Path, default=REPO_ROOT / "data" / "rtfs_m5_rich_metadata.csv")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "cache" / "rtfs_all_f6")
    parser.add_argument("--freeze-blocks", type=int, default=10, help="Freeze encoder layers 0..freeze_blocks-1.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    with args.metadata.open("r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_rows = [r for r in all_rows if r["split"] == "val"]
    test_rows = [r for r in all_rows if r["split"] == "test"]

    # Hard assertion: zero subject leakage across train/val/test
    s_tr = {r["subject_id"] for r in train_rows}
    s_va = {r["subject_id"] for r in val_rows}
    s_te = {r["subject_id"] for r in test_rows}
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "Subject leakage detected!"
    print(f"Splits verified leak-free: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    crops_by_sid = load_cached_crops(all_rows, args.cache_dir)

    def _flatten(rows_sub: List[dict]) -> Tuple[List[str], List[np.ndarray], np.ndarray]:
        sids, imgs, ys = [], [], []
        for r in rows_sub:
            y = 1 if r["label"] == "Deepfake" else 0
            for arr in crops_by_sid[r["sample_id"]]:
                sids.append(r["sample_id"])
                imgs.append(arr)
                ys.append(y)
        return sids, imgs, np.asarray(ys, dtype=np.int64)

    train_sids, train_imgs_clean, train_y = _flatten(train_rows)
    val_sids, val_imgs, _val_y_f = _flatten(val_rows)
    test_sids, test_imgs, _test_y_f = _flatten(test_rows)

    rng = random.Random(args.seed)
    print(f"Building forensic-augmented training view for {len(train_imgs_clean)} crops...")
    train_imgs_aug = [forensic_augment(arr, rng) for arr in train_imgs_clean]

    val_y_vid = np.asarray([1 if r["label"] == "Deepfake" else 0 for r in val_rows], dtype=int)

    results_models: Dict[str, Any] = {}
    val_scores_store: Dict[str, Dict[str, float]] = {}
    test_scores_store: Dict[str, Dict[str, float]] = {}
    emb_store: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    latencies: Dict[str, Dict[str, float]] = {}

    for arch, model_id in BACKBONES:
        print(f"\n==================== {arch.upper()} : {model_id} ====================")
        det = DeepfakeDetector(
            model_id=model_id,
            device="cpu",
            use_default_adapter=False,
            local_files_only=True,
        )
        fake_idx = int(det.fake_label_index)
        latencies[arch] = measure_cpu_latency(det, val_imgs[0])
        print(f"  CPU latency ({arch}): {latencies[arch]}")

        print("  Extracting frozen stem hidden states + zero-shot baselines...")
        tr_stem_clean, tr_emb_clean, _ = extract_stem_and_zeroshot(det, arch, args.freeze_blocks, train_imgs_clean)
        tr_stem_aug, tr_emb_aug, _ = extract_stem_and_zeroshot(det, arch, args.freeze_blocks, train_imgs_aug)
        va_stem, va_emb, va_zs_p = extract_stem_and_zeroshot(det, arch, args.freeze_blocks, val_imgs)
        te_stem, te_emb, te_zs_p = extract_stem_and_zeroshot(det, arch, args.freeze_blocks, test_imgs)

        # Store embeddings for dual-backbone linear probe ablation
        tr_emb_both = np.concatenate([tr_emb_clean, tr_emb_aug], axis=0)
        tr_y_both = np.concatenate([train_y, train_y], axis=0)
        emb_store[arch] = (tr_emb_both, va_emb, te_emb)

        # 1. Zero-shot evaluation
        zs_key = f"zeroshot_{arch}"
        va_zs_scores = aggregate_frame_probs(val_sids, va_zs_p)
        te_zs_scores = aggregate_frame_probs(test_sids, te_zs_p)
        zs_tau, _ = find_optimal_threshold(val_y_vid, np.asarray([va_zs_scores[r["sample_id"]] for r in val_rows]))
        val_scores_store[zs_key] = va_zs_scores
        test_scores_store[zs_key] = te_zs_scores
        results_models[zs_key] = {
            "model_id": model_id,
            "method": "Zero-shot Pretrained",
            "val_calibrated_tau": round(zs_tau, 4),
            "cpu_latency": latencies[arch],
            "val": evaluate_video_scores(val_rows, va_zs_scores, zs_tau, compute_ci=False),
            "test": evaluate_video_scores(test_rows, te_zs_scores, zs_tau, compute_ci=True),
        }
        print(
            f"  [{zs_key}] val_AUC={results_models[zs_key]['val']['roc_auc']} | "
            f"TEST_AUC={results_models[zs_key]['test']['roc_auc']} "
            f"(CI {results_models[zs_key]['test']['roc_auc_ci95']}) | "
            f"TEST_ACER={results_models[zs_key]['test']['at_val_tau']['acer']}"
        )

        # 2. Frozen Linear Probe ablation
        lp_key = f"linear_probe_{arch}"
        va_lp_scores, te_lp_scores, lp_tau, best_c = fit_frozen_linear_probe(
            tr_emb_both, tr_y_both, va_emb, val_sids, val_rows, te_emb, test_sids
        )
        val_scores_store[lp_key] = va_lp_scores
        test_scores_store[lp_key] = te_lp_scores
        results_models[lp_key] = {
            "model_id": model_id,
            "method": f"Frozen Embedding Linear Probe (C={best_c})",
            "val_calibrated_tau": round(lp_tau, 4),
            "cpu_latency": latencies[arch],
            "val": evaluate_video_scores(val_rows, va_lp_scores, lp_tau, compute_ci=False),
            "test": evaluate_video_scores(test_rows, te_lp_scores, lp_tau, compute_ci=True),
        }
        print(
            f"  [{lp_key}] val_AUC={results_models[lp_key]['val']['roc_auc']} | "
            f"TEST_AUC={results_models[lp_key]['test']['roc_auc']} "
            f"(CI {results_models[lp_key]['test']['roc_auc_ci95']}) | "
            f"TEST_ACER={results_models[lp_key]['test']['at_val_tau']['acer']}"
        )

        # 3. Top-2 Transformer Blocks + Final LayerNorm + Classifier Fine-Tuning
        ft_key = f"finetuned_top2_{arch}"
        print(f"  Fine-tuning top {12 - args.freeze_blocks} blocks + LayerNorm + Classifier ({arch})...")
        best_head, train_hist, va_ft_scores = train_top_blocks(
            hf_model=det._model,
            arch=arch,
            freeze_blocks=args.freeze_blocks,
            fake_idx=fake_idx,
            train_stem_clean=tr_stem_clean,
            train_stem_aug=tr_stem_aug,
            train_y_fake=train_y,
            val_stem=va_stem,
            val_sids=val_sids,
            val_rows=val_rows,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
        te_ft_scores = score_with_head(best_head, te_stem, test_sids, fake_idx)
        ft_tau, _ = find_optimal_threshold(val_y_vid, np.asarray([va_ft_scores[r["sample_id"]] for r in val_rows]))

        # Verify exact numerical parity after loading adapter back into standard HF model
        best_head.copy_into_hf_model(det._model)
        check_probs_full = det.predict_probs_raw(val_imgs[:4])
        with torch.inference_mode():
            check_logits, _ = best_head(va_stem[:4])
            check_probs_split = torch.softmax(check_logits, dim=-1)[:, fake_idx].tolist()
        max_parity_diff = max(abs(a - b) for a, b in zip(check_probs_full, check_probs_split))
        print(f"  Exact HF parity check max |diff| = {max_parity_diff:.6f}")

        # Save compact FP16 adapter checkpoint
        adapter_path = REPO_ROOT / "models" / f"m5_rtfs_{arch}_top2.pt"
        adapter_payload = best_head.export_adapter_state(
            model_id=model_id,
            id2label=det.id2label,
            fake_idx=fake_idx,
            val_tau=ft_tau,
        )
        torch.save(adapter_payload, adapter_path)
        adapter_mb = round(adapter_path.stat().st_size / (1024 * 1024), 2)
        print(f"  Saved adapter -> {adapter_path} ({adapter_mb} MB)")

        val_scores_store[ft_key] = va_ft_scores
        test_scores_store[ft_key] = te_ft_scores
        results_models[ft_key] = {
            "model_id": model_id,
            "method": f"Fine-tuned Top-{12 - args.freeze_blocks} Blocks + LN + Head",
            "adapter_path": str(adapter_path.relative_to(REPO_ROOT)),
            "adapter_size_mb": adapter_mb,
            "parity_max_diff": round(float(max_parity_diff), 6),
            "val_calibrated_tau": round(ft_tau, 4),
            "cpu_latency": latencies[arch],
            "train_history": train_hist,
            "val": evaluate_video_scores(val_rows, va_ft_scores, ft_tau, compute_ci=False),
            "test": evaluate_video_scores(test_rows, te_ft_scores, ft_tau, compute_ci=True),
        }
        print(
            f"  [{ft_key}] val_AUC={results_models[ft_key]['val']['roc_auc']} | "
            f"TEST_AUC={results_models[ft_key]['test']['roc_auc']} "
            f"(CI {results_models[ft_key]['test']['roc_auc_ci95']}) | "
            f"TEST_ACER={results_models[ft_key]['test']['at_val_tau']['acer']}"
        )

        # Release cached stem tensors before loading next backbone
        del tr_stem_clean, tr_stem_aug, va_stem, te_stem, det, best_head
        gc.collect()

    # 4. Dual-backbone Frozen Linear Probe ablation
    tr_dual = np.concatenate([emb_store["siglip2"][0], emb_store["vit"][0]], axis=1)
    va_dual = np.concatenate([emb_store["siglip2"][1], emb_store["vit"][1]], axis=1)
    te_dual = np.concatenate([emb_store["siglip2"][2], emb_store["vit"][2]], axis=1)
    tr_y_both = np.concatenate([train_y, train_y], axis=0)
    va_dlp, te_dlp, dlp_tau, dlp_c = fit_frozen_linear_probe(
        tr_dual, tr_y_both, va_dual, val_sids, val_rows, te_dual, test_sids
    )
    val_scores_store["linear_probe_dual"] = va_dlp
    test_scores_store["linear_probe_dual"] = te_dlp
    results_models["linear_probe_dual"] = {
        "model_id": "king1oo1/deepfake-model + prithivMLmods/Deep-Fake-Detector-v2-Model",
        "method": f"Dual-Backbone Frozen Linear Probe (C={dlp_c})",
        "val_calibrated_tau": round(dlp_tau, 4),
        "val": evaluate_video_scores(val_rows, va_dlp, dlp_tau, compute_ci=False),
        "test": evaluate_video_scores(test_rows, te_dlp, dlp_tau, compute_ci=True),
    }

    # 5. Ensembles (Zero-shot 50/50 & Fine-tuned 50/50)
    for ens_name, k1, k2, desc in [
        ("zeroshot_ensemble_50_50", "zeroshot_siglip2", "zeroshot_vit", "Zero-shot 50/50 Ensemble (SigLIP2 + ViT)"),
        ("finetuned_ensemble_50_50", "finetuned_top2_siglip2", "finetuned_top2_vit", "Fine-tuned Top-2 50/50 Ensemble (SigLIP2 + ViT)"),
    ]:
        va_ens = {r["sample_id"]: 0.5 * val_scores_store[k1][r["sample_id"]] + 0.5 * val_scores_store[k2][r["sample_id"]] for r in val_rows}
        te_ens = {r["sample_id"]: 0.5 * test_scores_store[k1][r["sample_id"]] + 0.5 * test_scores_store[k2][r["sample_id"]] for r in test_rows}
        ens_tau, _ = find_optimal_threshold(val_y_vid, np.asarray([va_ens[r["sample_id"]] for r in val_rows]))
        val_scores_store[ens_name] = va_ens
        test_scores_store[ens_name] = te_ens
        combined_b1 = latencies["siglip2"]["batch1_mean_ms"] + latencies["vit"]["batch1_mean_ms"]
        combined_b10 = latencies["siglip2"]["batch10_ms_per_frame"] + latencies["vit"]["batch10_ms_per_frame"]
        results_models[ens_name] = {
            "model_id": "king1oo1/deepfake-model + prithivMLmods/Deep-Fake-Detector-v2-Model",
            "method": desc,
            "val_calibrated_tau": round(ens_tau, 4),
            "cpu_latency": {
                "batch1_mean_ms": round(combined_b1, 2),
                "batch1_fps": round(1000.0 / combined_b1, 2),
                "batch10_ms_per_frame": round(combined_b10, 2),
                "batch10_fps": round(1000.0 / combined_b10, 2),
            },
            "val": evaluate_video_scores(val_rows, va_ens, ens_tau, compute_ci=False),
            "test": evaluate_video_scores(test_rows, te_ens, ens_tau, compute_ci=True),
        }
        print(
            f"\n  [{ens_name}] val_AUC={results_models[ens_name]['val']['roc_auc']} | "
            f"TEST_AUC={results_models[ens_name]['test']['roc_auc']} "
            f"(CI {results_models[ens_name]['test']['roc_auc_ci95']}) | "
            f"TEST_ACER={results_models[ens_name]['test']['at_val_tau']['acer']}"
        )

    # Save full E5 JSON report
    out_json = REPO_ROOT / "results" / "m5_e5_test_benchmark.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "protocol": {
                    "dataset": "dataset_RTFS_800_Subset (rtfs-10k)",
                    "split_seed": args.seed,
                    "n_train": len(train_rows),
                    "n_val": len(val_rows),
                    "n_test": len(test_rows),
                    "subject_overlap": 0,
                    "frames_per_video": 6,
                    "face_detector": "MTCNN (224x224 RGB, 25% margin)",
                    "freeze_blocks": args.freeze_blocks,
                    "epochs": args.epochs,
                },
                "models": results_models,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Save per-video test predictions CSV for full reproducibility
    out_csv = REPO_ROOT / "results" / "m5_e5_test_predictions.csv"
    pred_keys = list(test_scores_store.keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["sample_id", "label", "generator", "has_faceenhancer", "lighting", "gender"] + pred_keys
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in test_rows:
            row_out = {
                "sample_id": r["sample_id"],
                "label": r["label"],
                "generator": r["generator"],
                "has_faceenhancer": r["has_faceenhancer"],
                "lighting": r["lighting"],
                "gender": r["gender"],
            }
            for k in pred_keys:
                row_out[k] = round(test_scores_store[k][r["sample_id"]], 6)
            writer.writerow(row_out)

    print(f"\nSaved E5 test benchmark JSON -> {out_json}")
    print(f"Saved E5 test predictions CSV -> {out_csv}")


if __name__ == "__main__":
    main()
