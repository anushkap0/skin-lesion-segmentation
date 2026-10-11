"""
isicseg_core.py - framework-free parts of the ISIC 2018 segmentation study.

Data download and caching, augmentation, per-image metrics (incl. HD95 and the
ISIC thresholded Jaccard), statistics across seeds, and paper figures.
Only numpy / scipy / pandas / opencv / matplotlib are needed here, so this file
can be tested without a GPU or PyTorch.
"""
from __future__ import annotations

import glob
import json
import math
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage, stats

# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
ISIC_BASE = "https://isic-challenge-data.s3.amazonaws.com"
ISIC_URLS = {
    "isic2018_train_img": f"{ISIC_BASE}/2018/ISIC2018_Task1-2_Training_Input.zip",
    "isic2018_train_gt": f"{ISIC_BASE}/2018/ISIC2018_Task1_Training_GroundTruth.zip",
    "isic2018_val_img": f"{ISIC_BASE}/2018/ISIC2018_Task1-2_Validation_Input.zip",
    "isic2018_val_gt": f"{ISIC_BASE}/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
    "isic2018_test_img": f"{ISIC_BASE}/2018/ISIC2018_Task1-2_Test_Input.zip",
    "isic2018_test_gt": f"{ISIC_BASE}/2018/ISIC2018_Task1_Test_GroundTruth.zip",
    "isic2017_test_img": f"{ISIC_BASE}/2017/ISIC-2017_Test_v2_Data.zip",
    "isic2017_test_gt": f"{ISIC_BASE}/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip",
}
# which zip pairs make which dataset
DATASETS = {
    "isic2018_train": ("isic2018_train_img", "isic2018_train_gt"),
    "isic2018_val": ("isic2018_val_img", "isic2018_val_gt"),
    "isic2018_test": ("isic2018_test_img", "isic2018_test_gt"),
    "isic2017_test": ("isic2017_test_img", "isic2017_test_gt"),
}

MODEL_LABELS = {
    "unet_scratch": "U-Net (scratch)",
    "attunet_scratch": "Attention U-Net (scratch)",
    "unet_r34": "U-Net (ResNet-34)",
    "unetpp_r34": "UNet++ (ResNet-34)",
    "deeplabv3p_r34": "DeepLabV3+ (ResNet-34)",
    "segformer_b2": "SegFormer (MiT-B2)",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def download(url: str, dest: Path) -> Path:
    """Download with wget (resumable). Skips if the file is already complete."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and zipfile.is_zipfile(dest):
        return dest
    print(f"Downloading {url}")
    subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", str(dest), url], check=True)
    if not zipfile.is_zipfile(dest):
        raise RuntimeError(f"{dest} is not a valid zip; delete it and retry")
    return dest


def extract(zip_path: Path, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    marker = out_dir / ".extracted"
    if marker.exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    marker.touch()
    return out_dir


def pair_images_and_masks(img_dir: Path, mask_dir: Path):
    """Return sorted (image_id, image_path, mask_path) triples for ISIC folders."""
    masks = {}
    for p in glob.glob(str(Path(mask_dir) / "**" / "*_segmentation.png"), recursive=True):
        stem = Path(p).name.replace("_segmentation.png", "")
        masks[stem] = p
    pairs = []
    for p in glob.glob(str(Path(img_dir) / "**" / "*.jpg"), recursive=True):
        stem = Path(p).stem
        if "superpixels" in stem:
            continue
        if stem in masks:
            pairs.append((stem, p, masks[stem]))
    pairs.sort()
    return pairs


def pair_ph2(ph2_root: Path):
    """PH2 layout: <root>/IMDxxx/IMDxxx_Dermoscopic_Image/IMDxxx.bmp and IMDxxx_lesion/IMDxxx_lesion.bmp"""
    pairs = []
    for d in sorted(glob.glob(str(Path(ph2_root) / "**" / "IMD*_Dermoscopic_Image"), recursive=True)):
        case = Path(d).name.replace("_Dermoscopic_Image", "")
        img = Path(d) / f"{case}.bmp"
        msk = Path(d).parent / f"{case}_lesion" / f"{case}_lesion.bmp"
        if img.exists() and msk.exists():
            pairs.append((case, str(img), str(msk)))
    return pairs


def load_resize(pairs, size: int):
    """Read and resize. Images: INTER_AREA; masks: INTER_NEAREST, binarised at 127."""
    n = len(pairs)
    X = np.zeros((n, size, size, 3), np.uint8)
    Y = np.zeros((n, size, size), np.uint8)
    ids = []
    for i, (sid, ip, mp) in enumerate(pairs):
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise IOError(mp)
        X[i] = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        Y[i] = (cv2.resize(msk, (size, size), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
        ids.append(sid)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")
    return X, Y, np.array(ids)


def build_cache(name: str, raw_dir: Path, cache_dir: Path, size: int, ph2_root: str | None = None) -> Path:
    """Download (if needed), resize and save one dataset as <cache>/<name>_<size>.npz."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{name}_{size}.npz"
    if out.exists():
        return out
    if name == "ph2":
        if not ph2_root or not Path(ph2_root).exists():
            print("PH2 not found - skipped (download it manually, see notebook).")
            return None
        pairs = pair_ph2(Path(ph2_root))
    else:
        img_key, gt_key = DATASETS[name]
        dirs = []
        for key in (img_key, gt_key):
            z = download(ISIC_URLS[key], Path(raw_dir) / f"{key}.zip")
            dirs.append(extract(z, Path(raw_dir) / key))
        pairs = pair_images_and_masks(*dirs)
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found for {name}")
    print(f"{name}: {len(pairs)} pairs -> resizing to {size}")
    X, Y, ids = load_resize(pairs, size)
    np.savez(out, X=X, Y=Y, ids=ids)
    return out


def load_cache(path):
    d = np.load(path, allow_pickle=False)
    return d["X"], d["Y"], d["ids"]


def make_split(n: int, val_frac: float = 0.1, seed: int = 42):
    """Fixed train/validation split of the official training set (same for every run)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(n * val_frac))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


# ----------------------------------------------------------------------------
# 2. Augmentation (numpy/cv2 - no extra dependency)
# ----------------------------------------------------------------------------
def augment(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator):
    """img uint8 HxWx3, msk uint8 HxW -> augmented copies.

    Flips, 90-degree rotations, random affine (rotation +-30 deg, scale 0.9-1.1,
    shift +-6%), and colour jitter in HSV (brightness, contrast, saturation, hue).
    """
    h, w = msk.shape
    if rng.random() < 0.5:
        img, msk = img[:, ::-1], msk[:, ::-1]
    if rng.random() < 0.5:
        img, msk = img[::-1], msk[::-1]
    k = int(rng.integers(4))
    if k:
        img, msk = np.rot90(img, k), np.rot90(msk, k)
    img, msk = np.ascontiguousarray(img), np.ascontiguousarray(msk)
    if rng.random() < 0.7:
        ang = rng.uniform(-30, 30)
        sc = rng.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[:, 2] += rng.uniform(-0.0625, 0.0625, 2) * (w, h)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        msk = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.7:
        f = img.astype(np.float32)
        f = (f - f.mean()) * rng.uniform(0.8, 1.2) + f.mean() * rng.uniform(0.8, 1.2)
        f = np.clip(f, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-5, 6))) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.8, 1.2), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img, msk


def normalise(img_uint8: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 3xHxW, ImageNet-normalised."""
    f = img_uint8.astype(np.float32) / 255.0
    f = (f - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(f, (2, 0, 1)).copy()


# ----------------------------------------------------------------------------
# 3. Metrics (per image)
# ----------------------------------------------------------------------------
def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance in pixels. NaN when either mask is empty."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    dt_g = ndimage.distance_transform_edt(~sg)
    dt_p = ndimage.distance_transform_edt(~sp)
    d = np.concatenate([dt_g[sp], dt_p[sg]])
    return float(np.percentile(d, 95))


def image_metrics(pred: np.ndarray, gt: np.ndarray, with_hd: bool = True) -> dict:
    """pred, gt: binary HxW. Empty-vs-empty counts as perfect overlap."""
    p, g = pred.astype(bool), gt.astype(bool)
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    tn = float(p.size - tp - fp - fn)
    both_empty = (tp + fp + fn) == 0
    dice = 1.0 if both_empty else 2 * tp / (2 * tp + fp + fn)
    jac = 1.0 if both_empty else tp / (tp + fp + fn)
    prec = (1.0 if both_empty else 0.0) if (tp + fp) == 0 else tp / (tp + fp)
    rec = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    spec = 1.0 if (tn + fp) == 0 else tn / (tn + fp)
    out = {
        "accuracy": (tp + tn) / p.size,
        "dice": dice,
        "jaccard": jac,
        "thresholded_jaccard": jac if jac >= 0.65 else 0.0,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
    }
    if with_hd:
        out["hd95"] = hd95(p, g)
    return out


METRICS = ["accuracy", "dice", "jaccard", "thresholded_jaccard", "precision", "recall", "specificity", "hd95"]
HIGHER_IS_BETTER = {m: (m != "hd95") for m in METRICS}


def evaluate_masks(preds: np.ndarray, gts: np.ndarray, ids, with_hd: bool = True) -> pd.DataFrame:
    rows = []
    for i in range(len(preds)):
        r = image_metrics(preds[i], gts[i], with_hd)
        r["image_id"] = str(ids[i])
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. Results bookkeeping and statistics
# ----------------------------------------------------------------------------
def run_dir(results_dir, experiment, model, seed) -> Path:
    return Path(results_dir) / experiment / f"{model}__seed{seed}"


def is_done(results_dir, experiment, model, seed) -> bool:
    return (run_dir(results_dir, experiment, model, seed) / "done.json").exists()


def collect_per_image(results_dir, experiment, split) -> pd.DataFrame:
    """Stack every run's per-image metrics for one evaluation split."""
    frames = []
    for d in sorted(Path(results_dir, experiment).glob("*__seed*")):
        f = d / f"metrics_{split}.csv"
        if not (d / "done.json").exists() or not f.exists():
            continue
        model, seed = d.name.split("__seed")
        df = pd.read_csv(f)
        df["model"], df["seed"] = model, int(seed)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise(per_image: pd.DataFrame, metrics=METRICS, order=None) -> pd.DataFrame:
    """Mean over images per run, then mean, SD and 95% t-CI across seeds per model."""
    metrics = [m for m in metrics if m in per_image.columns]
    run = per_image.groupby(["model", "seed"])[metrics].mean().reset_index()
    rows = []
    for model, g in run.groupby("model"):
        n = len(g)
        row = {"model": model, "n_seeds": n}
        for m in metrics:
            v = g[m].to_numpy(float)
            mu = float(np.nanmean(v))
            sd = float(np.nanstd(v, ddof=1)) if n > 1 else float("nan")
            half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
            row[m], row[f"{m}_sd"], row[f"{m}_ci95"] = mu, sd, half
        rows.append(row)
    out = pd.DataFrame(rows)
    if order:
        out["__o"] = out["model"].map({m: i for i, m in enumerate(order)})
        out = out.sort_values("__o").drop(columns="__o")
    return out.reset_index(drop=True)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_tests(per_image: pd.DataFrame, reference: str, metric: str = "jaccard") -> pd.DataFrame:
    """Wilcoxon signed-rank test of each model vs `reference` on per-image `metric`
    (averaged over seeds first, so each image is one paired observation), Holm-corrected.
    Also reports a Friedman test across all models and the matched-pairs rank-biserial effect size."""
    piv = per_image.groupby(["image_id", "model"])[metric].mean().unstack("model").dropna()
    if reference not in piv.columns:
        raise KeyError(f"{reference} not in results")
    rows = []
    for m in piv.columns:
        if m == reference:
            continue
        diff = piv[m] - piv[reference]
        nz = diff[diff != 0]
        if len(nz) == 0:
            stat, p, rbc = float("nan"), 1.0, 0.0
        else:
            stat, p = stats.wilcoxon(piv[m], piv[reference], zero_method="wilcox")
            ranks = stats.rankdata(np.abs(nz))
            rbc = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
        rows.append({"model": m, "vs": reference, "metric": metric, "n_images": len(piv),
                     "mean_diff": float(diff.mean()), "median_diff": float(diff.median()),
                     "wilcoxon_W": float(stat), "p_value": float(p), "rank_biserial": rbc})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_value"])
        out["significant_0.05"] = out["p_holm"] < 0.05
    if piv.shape[1] >= 3:
        chi, pf = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
        out.attrs["friedman"] = {"chi2": float(chi), "p": float(pf), "k": piv.shape[1], "n": len(piv)}
    return out


def fmt_table(summary: pd.DataFrame, metrics, labels=None, decimals=4, bold_best=True, latex=False) -> str:
    """Markdown (or LaTeX) table of mean +- SD, best per column in bold."""
    labels = labels or {}
    metrics = [m for m in metrics if m in summary.columns]
    best = {}
    for m in metrics:
        col = summary[m]
        best[m] = col.idxmax() if HIGHER_IS_BETTER.get(m, True) else col.idxmin()
    names = {"accuracy": "Accuracy", "dice": "Dice (F1)", "jaccard": "Jaccard (IoU)",
             "thresholded_jaccard": "Thresholded Jaccard", "precision": "Precision",
             "recall": "Recall", "specificity": "Specificity", "hd95": "HD95 (px)"}
    header = ["Model"] + [names.get(m, m) for m in metrics]
    lines = []
    for i, r in summary.iterrows():
        cells = [labels.get(r["model"], r["model"])]
        for m in metrics:
            d = 2 if m == "hd95" else decimals
            sd = r.get(f"{m}_sd", float("nan"))
            s = f"{r[m]:.{d}f}" + ("" if pd.isna(sd) else f" ± {sd:.{d}f}")
            if bold_best and i == best[m]:
                s = (r"\textbf{" + s + "}") if latex else f"**{s}**"
            cells.append(s.replace("±", r"$\pm$") if latex else s)
        lines.append(cells)
    if latex:
        out = ["\\begin{tabular}{l" + "c" * len(metrics) + "}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
        out += [" & ".join(c) + r" \\" for c in lines]
        out += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(out)
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(c) + " |" for c in lines]
    return "\n".join(out)


# ----------------------------------------------------------------------------
# 5. Figures (matplotlib, 300 dpi, colour-blind-checked palette)
# ----------------------------------------------------------------------------
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "savefig.dpi": 300, "savefig.bbox": "tight"})
    return plt


def fig_box(per_image: pd.DataFrame, models, labels, metric, out_path, title=None):
    """Per-image distribution (seeds averaged) of one metric, one box per model."""
    plt = _plt()
    piv = per_image.groupby(["image_id", "model"])[metric].mean().unstack("model")
    models = [m for m in models if m in piv.columns]
    data = [piv[m].dropna().to_numpy() for m in models]
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(models) + 1.5), 3.6))
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "#0b0b0b"})
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(PALETTE[i % 8]); b.set_alpha(0.35); b.set_edgecolor(PALETTE[i % 8])
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(0).uniform(-0.18, 0.18, len(d)), d,
                   s=3, color=PALETTE[i % 8], alpha=0.25, linewidths=0)
        ax.scatter([i + 1], [d.mean()], marker="D", s=22, color="#0b0b0b", zorder=3)
    ax.set_xticks(range(1, len(models) + 1), [labels.get(m, m) for m in models], rotation=20, ha="right")
    ax.set_ylabel({"jaccard": "Jaccard (IoU) per image", "dice": "Dice per image", "hd95": "HD95 (px) per image"}.get(metric, metric))
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    fig.savefig(out_path); plt.close(fig)


def fig_curves(results_dir, experiment, models, labels, seed, out_path, key="val_dice"):
    """Validation curve of one seed per model on a shared axis."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, m in enumerate(models):
        f = run_dir(results_dir, experiment, m, seed) / "history.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f)
        ax.plot(h["epoch"], h[key], lw=1.8, color=PALETTE[i % 8], label=labels.get(m, m))
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch"); ax.set_ylabel({"val_dice": "Validation Dice", "val_loss": "Validation loss"}.get(key, key))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(out_path); plt.close(fig)


def fig_train_val(history_csv, out_path, title=None):
    """Training vs validation loss and Dice for one run (two panels, shared epochs)."""
    plt = _plt()
    h = pd.read_csv(history_csv)
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, (a, b, lab) in zip(axs, [("train_loss", "val_loss", "Loss"), ("train_dice", "val_dice", "Dice")]):
        ax.plot(h["epoch"], h[a], color=PALETTE[0], lw=1.8, label="Training")
        ax.plot(h["epoch"], h[b], color=PALETTE[1], lw=1.8, label="Validation")
        ax.set_xlabel("Epoch"); ax.set_ylabel(lab)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].legend(frameon=False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.savefig(out_path); plt.close(fig)


def fig_qualitative(X, Y, ids, preds_by_model: dict, labels, pick_ids, out_path):
    """Rows = cases, columns = image + GT + each model. Contours: GT white, prediction in model colour."""
    plt = _plt()
    models = list(preds_by_model)
    idx = [int(np.where(ids == p)[0][0]) for p in pick_ids]
    ncol = 2 + len(models)
    fig, axs = plt.subplots(len(idx), ncol, figsize=(1.6 * ncol, 1.6 * len(idx)))
    axs = np.atleast_2d(axs)
    for r, i in enumerate(idx):
        img = X[i]
        axs[r, 0].imshow(img)
        axs[r, 1].imshow(Y[i], cmap="gray")
        for c, m in enumerate(models):
            ax = axs[r, 2 + c]
            ax.imshow(img)
            ax.contour(Y[i], levels=[0.5], colors="white", linewidths=1.0)
            ax.contour(preds_by_model[m][i], levels=[0.5], colors=PALETTE[c % 8], linewidths=1.2)
            j = image_metrics(preds_by_model[m][i], Y[i], with_hd=False)["jaccard"]
            ax.text(4, 14, f"J={j:.2f}", color="white", fontsize=7, bbox=dict(fc="black", alpha=0.5, lw=0, pad=1))
        for ax in axs[r]:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for c, t in enumerate(["Image", "Ground truth"] + [labels.get(m, m) for m in models]):
        axs[0, c].set_title(t, fontsize=8)
    fig.savefig(out_path); plt.close(fig)
