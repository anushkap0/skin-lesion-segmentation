import json
from pathlib import Path

here = Path(__file__).parent
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md(r"""
# ISIC 2018 skin-lesion segmentation: benchmark and training-protocol study

This notebook produces the experiments for the paper *Training Protocol Matters: A Controlled Study of U-Net Optimisation for Dermoscopic Skin Lesion Segmentation*.

**Experiment A: architecture benchmark.** Six models are trained with one shared recipe, 5 seeds each: U-Net (scratch), Attention U-Net (scratch), U-Net with a ResNet-34 encoder, UNet++, DeepLabV3+ and SegFormer (MiT-B2). The recipe is AdamW with a cosine schedule, BCE + Dice loss, augmentation, early stopping on validation Dice with best-weight restoration, and flip test-time augmentation.

**Experiment B: protocol replication.** The scratch U-Net is trained under the paper's four protocols T1–T4, with 5 seeds each, to measure run-to-run variance.

**Evaluation.**
- Test sets: the official ISIC 2018 Task 1 test set (1,000 images), plus external testing on ISIC 2017 (600 images) and, optionally, PH2.
- Metrics: accuracy, Dice, Jaccard, the ISIC thresholded Jaccard, precision, recall, specificity and HD95.
- Statistics: mean ± SD, 95% CI, Wilcoxon signed-rank tests with Holm correction, and a Friedman test.

**Outputs.** Paper-ready tables (Markdown and LaTeX), figures at 300 dpi, and a `summary.md`.

### How to run
1. **Runtime → Change runtime type → GPU** (a T4 is enough; an A100 or L4 is about 3× faster).
2. Check the settings in the first code cell, then **Runtime → Run all**.
3. If Colab disconnects, reconnect and **Run all** again. Finished runs are skipped and unfinished runs resume from their last checkpoint, because results are written to Google Drive.
4. When it finishes, download `results_bundle.zip` (last cell) and send it back for the paper.

### Compute budget (T4, 256 × 256)
| Part | Runs | Approx. GPU time |
|---|---|---|
| Data download + caching (once) | – | 20–40 min |
| Experiment A, 6 models × 5 seeds | 30 | 15–25 h |
| Experiment B, 4 protocols × 5 seeds | 20 | 5–8 h |

Free Colab sessions end after a few hours, so the run will span several sessions. Kaggle Notebooks (30 free GPU hours a week) and Colab Pro also work. The notebook detects Kaggle and writes to `/kaggle/working`. To get first results sooner, set `SEEDS = [0, 1, 2]`; you can add seeds 3 and 4 later.
""")

code(r"""
# ===================== SETTINGS =====================
IMG_SIZE      = 256                      # training/evaluation resolution
SEEDS         = [0, 1, 2, 3, 4]          # use [0, 1, 2] for a faster first pass
MODELS        = ["unet_scratch", "attunet_scratch", "unet_r34", "unetpp_r34", "deeplabv3p_r34", "segformer_b2"]
RUN_BENCHMARK = True                     # Experiment A
RUN_PROTOCOLS = True                     # Experiment B (T1-T4 replication)
EXTERNAL      = ["isic2017_test"]        # external test sets downloaded automatically
PH2_ROOT      = None                     # e.g. "/content/drive/MyDrive/PH2Dataset" after manual download, else None
RUN_SMOKE_TEST = True                    # 1-2 minute check of every model before the long runs
# ====================================================
""")

code(r"""
import os, sys, subprocess
IN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
try:
    import google.colab  # noqa
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT = "/content/drive/MyDrive/isic2018_study"
    RAW = "/content/raw"                 # large zips stay on local disk, not Drive
elif IN_KAGGLE:
    PROJECT, RAW = "/kaggle/working/isic2018_study", "/kaggle/temp/raw"
else:
    PROJECT, RAW = os.path.abspath("isic2018_study"), os.path.abspath("raw")
CACHE, RESULTS, ANALYSIS = f"{PROJECT}/cache", f"{PROJECT}/results", f"{PROJECT}/analysis"
for d in (PROJECT, RAW, CACHE, RESULTS, ANALYSIS):
    os.makedirs(d, exist_ok=True)
CODE = f"{PROJECT}/code"; os.makedirs(CODE, exist_ok=True)
sys.path.insert(0, CODE)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch>=0.4.0", "tabulate"], check=True)
import torch, segmentation_models_pytorch as smp
print("torch", torch.__version__, "| smp", smp.__version__, "| CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "No GPU - set Runtime > Change runtime type > GPU"
""")

for name in ("isicseg_core.py", "isicseg_train.py", "isicseg_experiments.py"):
    src = (here.parent / "src" / name).read_text()
    md(f"`{name}`: written to the project folder so the analysis can be re-run later without the notebook.")
    code(f"%%writefile {{CODE}}/{name}\n" + src)

md("## 1. Download and cache the data\nThe official ISIC zips (about 11 GB) are downloaded once to local disk and resized to `IMG_SIZE`; the cache (under 1 GB) is kept in the project folder.")
code(r"""
import importlib, isicseg_core as core, isicseg_experiments as ex
importlib.reload(core); importlib.reload(ex)
data, eval_sets = ex.prepare_data(RAW, CACHE, IMG_SIZE, external=tuple(EXTERNAL), ph2_root=PH2_ROOT)
""")

md("## 2. Smoke test\nTrains every model for 2 epochs on 64 images, interrupting after epoch 1 and resuming, so problems surface in about 2 minutes instead of hours later.")
code(r"""
import shutil, numpy as np, isicseg_train as tr
importlib.reload(tr)
if RUN_SMOKE_TEST:
    tmp = "/tmp/smoke"; shutil.rmtree(tmp, ignore_errors=True)
    small = {**data, "train_idx": data["train_idx"][:64], "val_idx": data["val_idx"][:16]}
    sets = {k: tuple(a[:8] for a in v) for k, v in eval_sets.items()}
    for m in MODELS:
        cfg = dict(max_epochs=2, batch_size=8, ckpt_every=1, save_pred=True)
        d = f"{tmp}/{m}"
        assert tr.train_run(m, 0, {**cfg, "_stop_after": 1}, small, d, sets) is None   # simulated disconnect
        s = tr.train_run(m, 0, cfg, small, d, sets)                                    # resume and finish
        assert s["epochs_run"] == 2 and os.path.exists(f"{d}/metrics_isic2018_test.csv")
        print(f"OK {m}: {s['params']/1e6:.1f}M params")
    for p in ex.PROTOCOLS:
        s = tr.train_run("unet_scratch", 0, {**ex.PROTOCOLS[p], "max_epochs": 1}, small, f"{tmp}/proto_{p}", {"isic2018_test": sets["isic2018_test"]})
    shutil.rmtree(tmp, ignore_errors=True)
    print("Smoke test passed.")
""")

md("## 3. Experiment A: architecture benchmark\nResumable: re-running this cell skips finished runs. Seed-0 predictions are saved for the qualitative figure.")
code(r"""
if RUN_BENCHMARK:
    ex.run_benchmark(data, eval_sets, RESULTS, models=MODELS, seeds=SEEDS)
""")

md("## 4. Experiment B: replication of protocols T1–T4\nThe scratch U-Net under the four protocols of the original study (Adam 1e-4, BCE, no augmentation, early stopping on validation loss).")
code(r"""
if RUN_PROTOCOLS:
    ex.run_protocols(data, eval_sets, RESULTS, seeds=SEEDS)
""")

md("## 5. Analysis: tables, statistics and figures\nThis cell works on partial results too, so you can run it at any time.")
code(r"""
from IPython.display import Markdown, Image, display
importlib.reload(core); importlib.reload(ex)
summary = ex.analyse(RESULTS, ANALYSIS, data, eval_sets)
display(Markdown(open(summary).read()))
for f in ["box_jaccard_benchmark_isic2018_test.png", "qualitative_isic2018_test.png",
          "val_dice_curves_benchmark.png", "box_jaccard_protocol_isic2018_test.png"]:
    p = f"{ANALYSIS}/figures/{f}"
    if os.path.exists(p):
        display(Image(p, width=900))
""")

md("## 6. Bundle the results\nCreates `results_bundle.zip` with the analysis plus every run's per-image metrics and learning curves (no weights). Send this file back to update the paper.")
code(r"""
import zipfile, glob
bundle = f"{PROJECT}/results_bundle.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{ANALYSIS}/**/*", recursive=True):
        if os.path.isfile(f): z.write(f, os.path.relpath(f, PROJECT))
    for pat in ("*/*/metrics_*.csv", "*/*/history.csv", "*/*/done.json"):
        for f in glob.glob(f"{RESULTS}/{pat}"):
            z.write(f, os.path.relpath(f, PROJECT))
print(bundle, round(os.path.getsize(bundle) / 1e6, 1), "MB")
if IN_COLAB:
    from google.colab import files
    files.download(bundle)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
out = here / "ISIC2018_segmentation_study.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out)
