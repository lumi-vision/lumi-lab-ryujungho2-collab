"""Held-out clip evaluation for the research API."""
import hashlib
import math

import torch


def ensure_disjoint(training, validation):
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        return h.digest()

    known = {digest(path) for path in training}
    if any(digest(path) in known for path in validation):
        raise ValueError("Training and validation contain identical files")


def passes(report):
    if not isinstance(report, dict):
        return False
    pairs = (("recall", "min_recall"), ("fpr", "max_fpr"), ("hazard_mae", "max_hazard_mae"))
    for metric, limit in pairs:
        for key in (metric, limit):
            value = report.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                return False
    return (report.get("positive_pixels", 0) > 0 and report.get("negative_pixels", 0) > 0
            and report["recall"] >= report["min_recall"] and report["fpr"] <= report["max_fpr"]
            and report["hazard_mae"] <= report["max_hazard_mae"])


@torch.inference_mode()
def evaluate(model, extractor, clips, device, limits):
    model.eval()
    positive = negative = tp = fp = 0
    error = 0.0
    for video, clean, mask, fps in clips:
        aux_state = state = None
        for rgb, target, labels in zip(video, clean, mask):
            rgb = rgb.unsqueeze(0).to(device)
            aux, aux_state = extractor.step(rgb, aux_state, fps=fps)
            out = model.step(rgb, aux, state)
            state = out["state"]
            truth = labels.to(device) >= 0.5
            pred = out["mask"][0] > 0.5
            tp += int((pred & truth).sum())
            fp += int((pred & ~truth).sum())
            positive += int(truth.sum())
            negative += int((~truth).sum())
            error += float(((out["corrected"][0] - target.to(device)).abs() * truth).sum())
    return dict(clips=len(clips), positive_pixels=positive, negative_pixels=negative,
                recall=tp / positive if positive else None, fpr=fp / negative if negative else None,
                hazard_mae=error / (3 * positive) if positive else None,
                min_recall=limits[0], max_fpr=limits[1], max_hazard_mae=limits[2])
