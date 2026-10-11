"""
isicseg_train.py - models, training loop (resumable), inference with TTA.

Requires torch and segmentation-models-pytorch >= 0.4 (for SegFormer).
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import isicseg_core as core

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class AttentionGate(nn.Module):
    """Additive attention gate of Oktay et al. (2018)."""
    def __init__(self, c_skip, c_gate, c_int):
        super().__init__()
        self.wx = nn.Sequential(nn.Conv2d(c_skip, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.wg = nn.Sequential(nn.Conv2d(c_gate, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.psi = nn.Sequential(nn.Conv2d(c_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x, g):
        a = F.relu(self.wx(x) + self.wg(g), inplace=True)
        return x * self.psi(a)


class UNetScratch(nn.Module):
    """Classic U-Net (Ronneberger et al., 2015) with BatchNorm; optional attention gates."""
    def __init__(self, base=64, attention=False):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([DoubleConv(3, c[0])] + [DoubleConv(c[i], c[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(c[i] * 2, c[i]) for i in reversed(range(4))])
        self.att = nn.ModuleList([AttentionGate(c[i], c[i], max(c[i] // 2, 1)) for i in reversed(range(4))]) if attention else None
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            s = skips.pop()
            if self.att is not None:
                s = self.att[j](s, x)
            x = dec(torch.cat([s, x], 1))
        return self.head(x)


class SizeGuard(nn.Module):
    """Makes sure logits come back at input resolution (some decoders output 1/4)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y


MODEL_LABELS = core.MODEL_LABELS


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    w = "imagenet" if pretrained else None
    if name == "unet_scratch":
        net = UNetScratch(64, attention=False)
    elif name == "attunet_scratch":
        net = UNetScratch(64, attention=True)
    else:
        import segmentation_models_pytorch as smp
        if name == "unet_r34":
            net = smp.Unet("resnet34", encoder_weights=w, in_channels=3, classes=1)
        elif name == "unetpp_r34":
            net = smp.UnetPlusPlus("resnet34", encoder_weights=w, in_channels=3, classes=1)
        elif name == "deeplabv3p_r34":
            net = smp.DeepLabV3Plus("resnet34", encoder_weights=w, in_channels=3, classes=1)
        elif name == "segformer_b2":
            net = smp.Segformer("mit_b2", encoder_weights=w, in_channels=3, classes=1)
        else:
            raise ValueError(f"unknown model {name}")
    return SizeGuard(net)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class SegDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, idx, train: bool):
        self.X, self.Y, self.idx, self.train = X, Y, np.asarray(idx), train
        self.rng = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        if self.train and self.rng is None:
            info = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng((info.seed if info else torch.initial_seed()) % (2 ** 32))
        i = self.idx[k]
        img, msk = self.X[i], self.Y[i]
        if self.train:
            img, msk = core.augment(img, msk, self.rng)
        return torch.from_numpy(core.normalise(img)), torch.from_numpy(msk.astype(np.float32))[None]


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits.float())
    t = target.float()
    inter = (p * t).flatten(1).sum(1)
    den = p.flatten(1).sum(1) + t.flatten(1).sum(1)
    return 1 - ((2 * inter + eps) / (den + eps)).mean()


def make_loss(kind: str):
    bce = nn.BCEWithLogitsLoss()
    if kind == "bce":
        return lambda l, t: bce(l.float(), t)
    if kind == "dice":
        return soft_dice_loss
    if kind == "bce_dice":
        return lambda l, t: bce(l.float(), t) + soft_dice_loss(l, t)
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_probs(model, X, device, batch=32, tta=False, idx=None) -> np.ndarray:
    """Returns uint8 probabilities (0-255), shape N x H x W. TTA = mean of identity, h-flip, v-flip."""
    model.eval()
    idx = np.arange(len(X)) if idx is None else np.asarray(idx)
    out = np.zeros((len(idx),) + X.shape[1:3], np.uint8)
    use_amp = device.type == "cuda"
    for s in range(0, len(idx), batch):
        xb = torch.from_numpy(np.stack([core.normalise(X[i]) for i in idx[s:s + batch]])).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            p = torch.sigmoid(model(xb).float())
            if tta:
                p = p + torch.sigmoid(model(xb.flip(-1)).float()).flip(-1)
                p = p + torch.sigmoid(model(xb.flip(-2)).float()).flip(-2)
                p = p / 3
        out[s:s + len(xb)] = (p[:, 0].float().cpu().numpy() * 255).round().astype(np.uint8)
    return out


def hard_dice(probs_u8, gts):
    p = probs_u8 >= 128
    g = gts.astype(bool)
    inter = (p & g).reshape(len(p), -1).sum(1)
    den = p.reshape(len(p), -1).sum(1) + g.reshape(len(g), -1).sum(1)
    d = np.where(den == 0, 1.0, 2 * inter / np.maximum(den, 1))
    return float(d.mean())


# ----------------------------------------------------------------------------
# Training (one run = one model x one seed x one config), resumable per epoch
# ----------------------------------------------------------------------------
DEFAULT_CFG = dict(
    max_epochs=100, batch_size=16, lr=3e-4, weight_decay=1e-4, optimizer="adamw", scheduler="cosine",
    loss="bce_dice", augment=True, pretrained=True, monitor="val_dice", patience=15, restore_best=True,
    early_stopping=True, tta=True, num_workers=2, save_weights=False, save_pred=False, ckpt_every=5,
)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train_run(model_name, seed, cfg, data, out_dir, eval_sets, device=None, log_every=1):
    """
    data: dict with X, Y (training pool arrays) and train_idx, val_idx.
    eval_sets: dict name -> (X, Y, ids) evaluated after training.
    Writes history.csv, metrics_<set>.csv, preds_<set>.npz (optional), done.json.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        print(f"[skip] {out.name} already done"); return json.loads((out / "done.json").read_text())
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    X, Y = data["X"], data["Y"]
    model = build_model(model_name, cfg["pretrained"]).to(device)
    if cfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"], eta_min=cfg["lr"] * 0.01)
             if cfg["scheduler"] == "cosine" else None)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_fn = make_loss(cfg["loss"])
    g = torch.Generator(); g.manual_seed(seed)
    dl = torch.utils.data.DataLoader(SegDataset(X, Y, data["train_idx"], cfg["augment"]), batch_size=cfg["batch_size"],
                                     shuffle=True, drop_last=True, num_workers=cfg["num_workers"],
                                     pin_memory=use_amp, generator=g)
    vX, vY = X[data["val_idx"]], Y[data["val_idx"]]

    start, hist, best_score, best_state, best_epoch, wait = 1, [], None, None, 0, 0
    ck = out / "last.pt"
    s = None
    if ck.exists():  # resume after a disconnect
        try:
            s = torch.load(ck, map_location="cpu", weights_only=False)
        except Exception as e:  # checkpoint cut off by a disconnect: start this run again
            print(f"[warn] {ck} is unreadable ({type(e).__name__}); restarting {out.name} from epoch 1")
            ck.unlink(missing_ok=True)
            s = None
    if s is not None:
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
        if sched and s["sched"]:
            sched.load_state_dict(s["sched"])
        start, hist, best_score, best_epoch, wait = s["epoch"] + 1, s["hist"], s["best_score"], s["best_epoch"], s["wait"]
        best_state = s["best_state"]
        torch.set_rng_state(s["rng_cpu"]); g.set_state(s["rng_gen"])
        print(f"[resume] {out.name} from epoch {start}")

    maximise = cfg["monitor"] != "val_loss"
    for epoch in range(start, cfg["max_epochs"] + 1):
        model.train()
        ep_t0 = time.time()
        tl, td, nb = 0.0, 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
            loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                td += 1 - soft_dice_loss((logits.float() > 0).float() * 40 - 20, yb).item()
            tl += loss.item(); nb += 1
        if sched:
            sched.step()
        # validation
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(vX), 32):
                xb = torch.from_numpy(np.stack([core.normalise(im) for im in vX[s:s + 32]])).to(device)
                yb = torch.from_numpy(vY[s:s + 32].astype(np.float32))[:, None].to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    lg = model(xb)
                vl += loss_fn(lg, yb).item() * len(xb); vn += len(xb)
        vprob = predict_probs(model, vX, device, tta=False)
        row = {"epoch": epoch, "train_loss": tl / max(nb, 1), "train_dice": td / max(nb, 1),
               "val_loss": vl / vn, "val_dice": hard_dice(vprob, vY), "lr": opt.param_groups[0]["lr"],
               "seconds": time.time() - ep_t0}
        hist.append(row)
        score = row[cfg["monitor"]]
        improved = best_score is None or (score > best_score if maximise else score < best_score)
        if improved:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch % log_every == 0:
            print(f"{out.name} ep {epoch:3d} | train {row['train_loss']:.4f}/{row['train_dice']:.4f} "
                  f"| val {row['val_loss']:.4f}/{row['val_dice']:.4f} | best {best_epoch}")
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        stopping = (cfg["early_stopping"] and wait >= cfg["patience"]) or bool(cfg.get("_stop_after") and epoch >= cfg["_stop_after"])
        if epoch % cfg["ckpt_every"] == 0 or stopping:
          torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                      "sched": sched.state_dict() if sched else None, "epoch": epoch, "hist": hist,
                      "best_score": best_score, "best_epoch": best_epoch, "wait": wait, "best_state": best_state,
                      "rng_cpu": torch.get_rng_state(), "rng_gen": g.get_state()}, ck.with_suffix(".tmp"))
          os.replace(ck.with_suffix(".tmp"), ck)  # atomic: a disconnect never leaves a half-written last.pt
        if cfg.get("_stop_after") and epoch >= cfg["_stop_after"]:  # used by the smoke test to simulate a disconnect
            print(f"{out.name}: stopping session after epoch {epoch} (resume test)")
            return None
        if cfg["early_stopping"] and wait >= cfg["patience"]:
            print(f"{out.name}: early stop at epoch {epoch} (best {best_epoch})")
            break
    epochs_run = hist[-1]["epoch"]
    if cfg["restore_best"] and best_state is not None:
        model.load_state_dict(best_state)
    train_seconds = float(sum(h["seconds"] for h in hist))

    # evaluation
    summary = {"model": model_name, "seed": seed, "cfg": cfg, "epochs_run": epochs_run, "best_epoch": best_epoch,
               "best_val": best_score, "params": count_params(model), "train_seconds": train_seconds}
    for name, (eX, eY, eids) in eval_sets.items():
        t1 = time.time()
        probs = predict_probs(model, eX, device, tta=cfg["tta"])
        dt = time.time() - t1
        preds = (probs >= 128).astype(np.uint8)
        df = core.evaluate_masks(preds, eY, eids)
        df.to_csv(out / f"metrics_{name}.csv", index=False)
        summary[f"{name}_ms_per_image"] = 1000 * dt / len(eX)
        summary[name] = {m: float(df[m].mean()) for m in core.METRICS}
        if cfg["save_pred"]:
            np.savez_compressed(out / f"preds_{name}.npz", probs=probs, ids=eids)
        print(f"  {name}: Dice {summary[name]['dice']:.4f}  Jaccard {summary[name]['jaccard']:.4f}")
    if cfg["save_weights"]:
        torch.save(model.state_dict(), out / "best_weights.pt")
    (out / "done.json").write_text(json.dumps(summary, indent=2, default=str))
    ck.unlink(missing_ok=True)
    return summary
