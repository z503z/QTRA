"""Training utilities: stage freezing, losses and validation-only records."""
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .evaluation import autocast_context
from .metrics import DiceLoss, prepare_batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def configure_stage(model, cfg, epoch):
    """Return optimizer groups and active module names for the current stage."""
    stage = cfg["stage"]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups = []
    active = []

    def enable(name, lr):
        module = model.get_submodule(name)
        params = list(module.parameters())
        for parameter in params:
            parameter.requires_grad_(True)
        groups.append({"params": params, "lr": lr, "name": name})
        active.append(name)

    lr = cfg["lr"]
    phase = stage
    if stage in ("baseline", "cdvqa"):
        for name, module in model.named_children():
            if name == "text_encoder" and cfg.get("freeze_text", True):
                continue
            enable(name, lr)
    elif stage == "qtra":
        later = epoch >= cfg["warmup_epochs"]
        phase = "qtra_unfreeze" if later else "qtra_frozen"
        enable("change_encoder.shared_qra", cfg["adapter_lr"] if later else cfg["adapter_warmup_lr"])
        for name in ("answer_head", "channel_gate", "mask_decoder"):
            enable(name, cfg["head_lr"])
        if later:
            enable("change_encoder.x3d.blocks.4", cfg["backbone_lr"])
    elif stage == "global":
        enable("answer_calibration", lr)
    elif stage == "grounding":
        for name in ("mask_decoder", "dynamic_mask_decoder", "dual_gate"):
            enable(name, lr)
    elif stage == "evidence":
        enable("mask_guided_answer", lr)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return phase, groups, active


def set_train_mode(model, active, stage):
    # Freeze running statistics/dropout in inactive modules, not just weights.
    model.eval()
    for name in active:
        model.get_submodule(name).train()
    if stage == "qtra":
        model.change_encoder.x3d.eval()


def make_optimizer(cfg, groups):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(groups, lr=cfg["lr"])
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")


def train_epoch(model, loader, optimizer, scaler, device, task, cfg, active, epoch):
    set_train_mode(model, active, cfg["stage"])
    accumulator = cfg.get("grad_accum_steps", 1)
    precision = cfg.get("precision", "fp32")
    dice = DiceLoss()
    totals = {"loss": 0.0, "answer_loss": 0.0, "mask_loss": 0.0}
    correct = samples = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc=f"train {epoch}")):
        if task == "qag":
            image1, image2, question, _, labels, mask, _ = prepare_batch(batch, device)
        else:
            image1, image2, question, _, labels, _ = batch
            image1, image2, question, labels = [
                tensor.to(device) for tensor in (image1, image2, question, labels)
            ]
        # Only the global refinement can omit mask computation during training.
        if hasattr(model, "skip_mask_forward"):
            model.skip_mask_forward = cfg["stage"] == "global"
        with autocast_context(device, precision):
            outputs = model(image1, image2, question)
            logits = outputs["answer_logits"] if task == "qag" else outputs
            answer_loss = F.cross_entropy(logits.float(), labels)
            mask_loss = logits.new_zeros(())
            if task == "qag" and cfg["lambda_mask"] > 0:
                masks = outputs["mask_logits"].float()
                if masks.shape[-2:] != mask.shape[-2:]:
                    masks = F.interpolate(masks, size=mask.shape[-2:], mode="bilinear", align_corners=False)
                mask_loss = F.binary_cross_entropy_with_logits(masks, mask) + dice(masks, mask)
            loss = cfg["lambda_answer"] * answer_loss + cfg["lambda_mask"] * mask_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {step}")
        # Retain the original accumulation denominator, including a final partial group.
        scaler.scale(loss / accumulator).backward()
        if (step + 1) % accumulator == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        totals["loss"] += loss.item()
        totals["answer_loss"] += answer_loss.item()
        totals["mask_loss"] += mask_loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        samples += labels.numel()
    if hasattr(model, "skip_mask_forward"):
        model.skip_mask_forward = False
    if samples == 0:
        raise ValueError("Empty training split.")
    return {**{key: value / len(loader) for key, value in totals.items()}, "oa": correct / samples}


def validation_scores(metrics):
    scores = {"val_oa": metrics["oa"], "val_aa": metrics["aa"]}
    scores["val_answer"] = 0.5 * (metrics["oa"] + metrics["aa"])
    if "miou" in metrics:
        scores["val_miou"] = metrics["miou"]
        scores["val_grounding"] = 0.5 * (metrics["miou"] + metrics["oiou"])
    return scores


def epoch_record(epoch, phase, optimizer, training, validation):
    record = {"epoch": epoch, "phase": phase}
    record.update({f"lr_{group['name']}": group["lr"] for group in optimizer.param_groups})
    record.update({f"train_{key}": value for key, value in training.items()})
    record.update({f"val_{key}": validation[key] for key in ("oa", "aa", "miou", "oiou") if key in validation})
    return record
