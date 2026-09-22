"""Benchmark metric helpers; formulas retained from the experiment code."""
import torch
import torch.nn as nn
import torch.nn.functional as F

CANONICAL_QAG_TYPES = ("change_ratio", "change_or_not", "change_to_what", "increase_or_not", "decrease_or_not", "smallest_change", "largest_change", "change_from_what")

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, mask_logits, target):
        pred = torch.sigmoid(mask_logits)
        pred = pred.flatten(1)
        target = target.flatten(1)
        intersection = (pred * target).sum(dim=1)
        pred_sum = pred.sum(dim=1)
        target_sum = target.sum(dim=1)
        dice = 1.0 - (2.0 * intersection + self.eps) / (pred_sum + target_sum + self.eps)
        return dice.mean()


def prepare_batch(batch, device):
    img1, img2, q_str, type_str, answer_vec, mask, sample_id = batch

    img1 = img1.to(device)
    img2 = img2.to(device)
    q_str = q_str.to(device)

    answer_vec = answer_vec.to(device)
    if answer_vec.ndim == 2:
        answer_label = answer_vec.argmax(dim=1).long()
    else:
        answer_label = answer_vec.long()

    mask = mask.to(device).float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] > 1:
        mask = mask[:, :1]
    mask = (mask > 0.5).float()

    return img1, img2, q_str, type_str, answer_label, mask, sample_id


def unpack_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs["answer_logits"], outputs["mask_logits"]
    if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        return outputs[0], outputs[1]
    raise TypeError("Model output must be a dict or a tuple/list with answer and mask logits.")


def normalize_question_type(type_name):
    type_name = str(type_name)
    if type_name == "change_ratio_types":
        return "change_ratio"
    return type_name


def make_type_confusion():
    return {t: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "count": 0} for t in CANONICAL_QAG_TYPES}


def calc_mask_confusion(mask_logits, target, threshold):
    if mask_logits.shape[-2:] != target.shape[-2:]:
        mask_logits = F.interpolate(mask_logits, size=target.shape[-2:], mode="bilinear", align_corners=False)

    pred = (torch.sigmoid(mask_logits) >= threshold)
    target = (target >= 0.5)

    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    tp = (pred_flat & target_flat).sum(dim=1).double()
    fp = (pred_flat & ~target_flat).sum(dim=1).double()
    fn = (~pred_flat & target_flat).sum(dim=1).double()

    return tp, fp, fn


def add_mask_confusion_by_type(type_confusion, type_str, tp, fp, fn):
    unknown_types = set()
    tp_cpu = tp.detach().cpu()
    fp_cpu = fp.detach().cpu()
    fn_cpu = fn.detach().cpu()
    for raw_type, sample_tp, sample_fp, sample_fn in zip(type_str, tp_cpu, fp_cpu, fn_cpu):
        type_name = normalize_question_type(raw_type)
        if type_name not in type_confusion:
            unknown_types.add(type_name)
            continue
        stats = type_confusion[type_name]
        stats["tp"] += sample_tp.item()
        stats["fp"] += sample_fp.item()
        stats["fn"] += sample_fn.item()
        stats["count"] += 1
    return unknown_types


def summarize_mask_confusion(type_confusion, desc):
    type_iou = {}
    missing_types = []
    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0

    for type_name in CANONICAL_QAG_TYPES:
        stats = type_confusion[type_name]
        tp_total += stats["tp"]
        fp_total += stats["fp"]
        fn_total += stats["fn"]
        denom = stats["tp"] + stats["fp"] + stats["fn"]
        type_iou[type_name] = stats["tp"] / denom if denom > 0 else 0.0
        if stats["count"] == 0:
            missing_types.append(type_name)

    if missing_types:
        print(f"WARNING: {desc} missing question types for fixed-8 mIoU: {missing_types}")

    miou = sum(type_iou[type_name] for type_name in CANONICAL_QAG_TYPES) / len(CANONICAL_QAG_TYPES)
    oiou = tp_total / max(tp_total + fp_total + fn_total, 1.0)
    return miou, oiou, type_iou, missing_types
