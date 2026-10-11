"""
isicseg_experiments.py - experiment definitions, runners and the analysis stage.

The analysis functions need only numpy/pandas/scipy/matplotlib; the runners
import torch lazily, so analysis can be re-run anywhere from the results folder.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import isicseg_core as core

# ----------------------------------------------------------------------------
# Experiment definitions
# ----------------------------------------------------------------------------
BENCHMARK_MODELS = ["unet_scratch", "attunet_scratch", "unet_r34", "unetpp_r34", "deeplabv3p_r34", "segformer_b2"]
SEEDS = [0, 1, 2, 3, 4]

# Experiment A: every model, same modern recipe (AdamW, cosine LR, BCE+Dice, augmentation,
# early stopping on validation Dice with best-weight restoration, flip TTA at test time).
BENCHMARK_CFG = dict(max_epochs=100, batch_size=16, lr=3e-4, weight_decay=1e-4, optimizer="adamw",
                     scheduler="cosine", loss="bce_dice", augment=True, pretrained=True,
                     monitor="val_dice", patience=15, restore_best=True, early_stopping=True, tta=True)

# Experiment B: replication of the four training protocols of the original study on the
# scratch U-Net (Adam 1e-4, constant LR, BCE, no augmentation, early stopping on val loss).
_PROTO_BASE = dict(optimizer="adam", lr=1e-4, scheduler="none", loss="bce", augment=False, pretrained=False,
                   monitor="val_loss", tta=False)
PROTOCOLS = {
    "T1": {**_PROTO_BASE, "max_epochs": 5, "batch_size": 4, "early_stopping": False, "restore_best": False, "patience": 0},
    "T2": {**_PROTO_BASE, "max_epochs": 100, "batch_size": 8, "early_stopping": True, "patience": 20, "restore_best": False},
    "T3": {**_PROTO_BASE, "max_epochs": 25, "batch_size": 8, "early_stopping": True, "patience": 10, "restore_best": True},
    "T4": {**_PROTO_BASE, "max_epochs": 100, "batch_size": 16, "early_stopping": True, "patience": 10, "restore_best": True},
}
PROTOCOL_LABELS = {"T1": "T1: 5 ep, bs 4", "T2": "T2: bs 8, P20, last weights",
                   "T3": "T3: 25 ep, bs 8, P10, best weights", "T4": "T4: bs 16, P10, best weights"}

TABLE_METRICS = ["accuracy", "dice", "jaccard", "thresholded_jaccard", "precision", "recall", "hd95"]


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------
def prepare_data(raw_dir, cache_dir, size=256, external=("isic2017_test",), ph2_root=None, val_frac=0.1):
    """Builds/loads caches. Returns (train data dict, eval_sets dict)."""
    paths = {n: core.build_cache(n, raw_dir, cache_dir, size) for n in ("isic2018_train", "isic2018_val", "isic2018_test")}
    for n in external:
        paths[n] = core.build_cache(n, raw_dir, cache_dir, size)
    if ph2_root:
        p = core.build_cache("ph2", raw_dir, cache_dir, size, ph2_root=ph2_root)
        if p:
            paths["ph2"] = p
    X, Y, ids = core.load_cache(paths["isic2018_train"])
    vX, vY, vids = core.load_cache(paths["isic2018_val"])
    # pool = official training (2,594) + official validation (100); fixed 90/10 split for early stopping
    PX, PY, Pids = np.concatenate([X, vX]), np.concatenate([Y, vY]), np.concatenate([ids, vids])
    tr, va = core.make_split(len(PX), val_frac=val_frac, seed=42)
    data = {"X": PX, "Y": PY, "ids": Pids, "train_idx": tr, "val_idx": va}
    eval_sets = {"isic2018_test": core.load_cache(paths["isic2018_test"])}
    for n in list(external) + (["ph2"] if "ph2" in paths else []):
        eval_sets[n] = core.load_cache(paths[n])
    print(f"train {len(tr)} | val {len(va)} | " + " | ".join(f"{k} {len(v[0])}" for k, v in eval_sets.items()))
    return data, eval_sets


# ----------------------------------------------------------------------------
# Runners (resumable: finished runs are skipped, unfinished runs resume from last epoch)
# ----------------------------------------------------------------------------
def run_benchmark(data, eval_sets, results_dir, models=BENCHMARK_MODELS, seeds=SEEDS, cfg_overrides=None, pred_seed=0):
    import isicseg_train as tr
    out = []
    for seed in seeds:  # seed-major order: one full set of models first, so partial results are usable
        for m in models:
            cfg = {**BENCHMARK_CFG, **(cfg_overrides or {}), "save_pred": seed == pred_seed}
            d = core.run_dir(results_dir, "benchmark", m, seed)
            out.append(tr.train_run(m, seed, cfg, data, d, eval_sets))
    return out


def run_protocols(data, eval_sets, results_dir, protocols=tuple(PROTOCOLS), seeds=SEEDS, pred_seed=0):
    import isicseg_train as tr
    sets = {"isic2018_test": eval_sets["isic2018_test"]}
    out = []
    for seed in seeds:
        for p in protocols:
            cfg = {**PROTOCOLS[p], "save_pred": seed == pred_seed}
            d = core.run_dir(results_dir, "protocol", p, seed)
            out.append(tr.train_run("unet_scratch", seed, cfg, data, d, sets))
    return out


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def _done(results_dir, experiment):
    rows = []
    for f in sorted(Path(results_dir, experiment).glob("*__seed*/done.json")):
        s = json.loads(f.read_text())
        s["run"] = f.parent.name.split("__seed")[0]
        rows.append(s)
    return rows


def efficiency_table(results_dir, experiment, labels):
    rows = []
    for s in _done(results_dir, experiment):
        rows.append({"model": s["run"], "params_M": s["params"] / 1e6, "epochs_run": s["epochs_run"],
                     "best_epoch": s["best_epoch"], "train_minutes": s.get("train_seconds", np.nan) / 60,
                     "ms_per_image_test": s.get("isic2018_test_ms_per_image", np.nan)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).groupby("model").agg(["mean", "std"]).round(3)
    df.columns = [f"{a}_{b}" for a, b in df.columns]
    df.insert(0, "label", [labels.get(m, m) for m in df.index])
    return df.reset_index()


def analyse(results_dir, out_dir, data=None, eval_sets=None):
    """Writes tables (md/tex/csv), statistical tests, figures and summary.md into out_dir."""
    results_dir, out = Path(results_dir), Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(exist_ok=True); (out / "stats").mkdir(exist_ok=True)
    md = ["# Results summary\n"]

    for exp, order, labels in [("benchmark", BENCHMARK_MODELS, core.MODEL_LABELS),
                               ("protocol", list(PROTOCOLS), PROTOCOL_LABELS)]:
        if not (results_dir / exp).exists():
            continue
        splits = sorted({f.stem.replace("metrics_", "") for f in (results_dir / exp).glob("*/metrics_*.csv")})
        for split in splits:
            pim = core.collect_per_image(results_dir, exp, split)
            if pim.empty:
                continue
            summ = core.summarise(pim, TABLE_METRICS, order=order)
            tag = f"{exp}_{split}"
            summ.to_csv(out / "tables" / f"{tag}.csv", index=False)
            t_md = core.fmt_table(summ, TABLE_METRICS, labels)
            (out / "tables" / f"{tag}.md").write_text(t_md)
            (out / "tables" / f"{tag}.tex").write_text(core.fmt_table(summ, TABLE_METRICS, labels, latex=True))
            seeds = ", ".join(f"{labels.get(m, m)}: {n}" for m, n in zip(summ["model"], summ["n_seeds"]))
            md += [f"## {exp} - {split}\n", f"Mean ± SD across seeds (seeds per model: {seeds}).\n", t_md, ""]
            md.append("| Model | Jaccard 95% CI |\n|---|---|")
            md += [f"| {labels.get(r.model, r.model)} | {r.jaccard:.4f} ± {r.jaccard_ci95:.4f} |" for r in summ.itertuples()]
            md.append("")
            # statistics: vs best model and vs the first model in the order (baseline)
            present = [m for m in order if m in set(pim["model"])]
            if len(present) >= 2:
                best = summ.loc[summ["jaccard"].idxmax(), "model"]
                for ref in dict.fromkeys([best, present[0]]):
                    for metric in ("jaccard", "dice"):
                        tst = core.paired_tests(pim, ref, metric)
                        tst = tst.assign(_o=tst["model"].map({m: i for i, m in enumerate(order)})).sort_values("_o").drop(columns="_o")
                        tst.to_csv(out / "stats" / f"{tag}_wilcoxon_{metric}_vs_{ref}.csv", index=False)
                        if metric == "jaccard":
                            md.append(f"**Wilcoxon signed-rank vs {labels.get(ref, ref)} (per-image Jaccard, Holm-corrected)**\n")
                            md.append("| Model | Mean diff | p (Holm) | Rank-biserial r |\n|---|---|---|---|")
                            md += [f"| {labels.get(r.model, r.model)} | {r.mean_diff:+.4f} | {r.p_holm:.2e} | {r.rank_biserial:+.3f} |"
                                   for r in tst.itertuples()]
                            fr = tst.attrs.get("friedman")
                            if fr:
                                md.append(f"\nFriedman test across {fr['k']} models, {fr['n']} images: "
                                          f"χ² = {fr['chi2']:.1f}, p = {fr['p']:.2e}.")
                                (out / "stats" / f"{tag}_friedman_jaccard.json").write_text(json.dumps(fr, indent=2))
                            md.append("")
                core.fig_box(pim, order, labels, "jaccard", out / "figures" / f"box_jaccard_{tag}.png")
                if "hd95" in pim:
                    core.fig_box(pim, order, labels, "hd95", out / "figures" / f"box_hd95_{tag}.png")
        eff = efficiency_table(results_dir, exp, labels)
        if len(eff):
            eff.to_csv(out / "tables" / f"{exp}_efficiency.csv", index=False)
            md += [f"## {exp} - training and inference cost\n", eff.to_markdown(index=False), ""]
        # learning curves (seed 0)
        present = [m for m in order if core.run_dir(results_dir, exp, m, 0).joinpath("history.csv").exists()]
        if present:
            core.fig_curves(results_dir, exp, present, labels, 0, out / "figures" / f"val_dice_curves_{exp}.png", "val_dice")
            core.fig_curves(results_dir, exp, present, labels, 0, out / "figures" / f"val_loss_curves_{exp}.png", "val_loss")
            for m in present:
                core.fig_train_val(core.run_dir(results_dir, exp, m, 0) / "history.csv",
                                   out / "figures" / f"train_val_{exp}_{m}.png", labels.get(m, m))

    # qualitative figure from saved seed-0 predictions on the ISIC 2018 test set
    if eval_sets is not None and "isic2018_test" in eval_sets:
        X, Y, ids = eval_sets["isic2018_test"]
        preds = {}
        for m in BENCHMARK_MODELS:
            f = core.run_dir(results_dir, "benchmark", m, 0) / "preds_isic2018_test.npz"
            if f.exists():
                z = np.load(f)
                preds[m] = (z["probs"] >= 128).astype(np.uint8)
        if preds:
            ref = next(iter(preds))
            j = np.array([core.image_metrics(preds[ref][i], Y[i], with_hd=False)["jaccard"] for i in range(len(Y))])
            order_idx = np.argsort(j)
            picks = [order_idx[-1], order_idx[int(len(j) * 0.75)], order_idx[len(j) // 2],
                     order_idx[int(len(j) * 0.1)], order_idx[0]]
            core.fig_qualitative(X, Y, ids, preds, core.MODEL_LABELS, [ids[i] for i in picks],
                                 out / "figures" / "qualitative_isic2018_test.png")
            md.append("Qualitative figure rows: best, 75th, 50th, 10th percentile and worst case by "
                      f"{core.MODEL_LABELS.get(ref, ref)} Jaccard.\n")

    (out / "summary.md").write_text("\n".join(md))
    print(f"Analysis written to {out}")
    return out / "summary.md"
