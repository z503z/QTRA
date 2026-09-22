"""Read-only evaluation shared by validation and the standalone evaluator."""
from contextlib import nullcontext

import torch
from tqdm import tqdm

from .metrics import (
    CANONICAL_QAG_TYPES, add_mask_confusion_by_type, calc_mask_confusion,
    make_type_confusion, normalize_question_type, prepare_batch,
    summarize_mask_confusion,
)


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("Mixed precision requires CUDA; use fp32 on CPU.")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(model, loader, device, task, threshold=0.35, precision="fp32", desc="eval"):
    model.eval()
    correct = total = 0
    type_correct = {t: 0 for t in CANONICAL_QAG_TYPES} if task == "qag" else {}
    type_total = dict(type_correct)
    confusion = make_type_confusion()
    for batch in tqdm(loader, desc=desc):
        if task == "qag":
            image1, image2, question, types, labels, mask, _ = prepare_batch(batch, device)
        else:
            image1, image2, question, types, labels, _ = batch
            image1, image2, question, labels = [
                tensor.to(device) for tensor in (image1, image2, question, labels)
            ]
        with autocast_context(device, precision):
            outputs = model(image1, image2, question)
        logits = outputs["answer_logits"] if task == "qag" else outputs
        prediction = logits.argmax(dim=1)
        correct += (prediction == labels).sum().item()
        total += labels.numel()
        for name, pred, label in zip(types, prediction.cpu(), labels.cpu()):
            name = normalize_question_type(name) if task == "qag" else str(name)
            if task == "qag" and name not in type_total:
                raise ValueError(f"Unknown question type: {name}")
            type_total[name] = type_total.get(name, 0) + 1
            type_correct[name] = type_correct.get(name, 0) + int(pred == label)
        if task == "qag":
            tp, fp, fn = calc_mask_confusion(outputs["mask_logits"], mask, threshold)
            add_mask_confusion_by_type(confusion, types, tp, fp, fn)
    if total == 0:
        raise ValueError("Cannot evaluate an empty split.")
    type_acc = {key: type_correct[key] / max(count, 1) for key, count in type_total.items()}
    metrics = {
        "oa": correct / total,
        "aa": sum(type_acc.values()) / len(type_acc),
        "type_acc": type_acc,
        "type_total": type_total,
        "num_samples": total,
    }
    if task == "qag":
        miou, oiou, type_iou, missing = summarize_mask_confusion(confusion, desc)
        metrics.update(miou=miou, oiou=oiou, mask_type_iou=type_iou, missing_types=missing)
    return metrics
