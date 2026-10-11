"""Command-line runner: python run_experiments.py --project ./isic2018_study [--seeds 0 1 2 3 4]"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import isicseg_experiments as ex

p = argparse.ArgumentParser()
p.add_argument("--project", default="isic2018_study")
p.add_argument("--raw", default="raw")
p.add_argument("--size", type=int, default=256)
p.add_argument("--seeds", type=int, nargs="+", default=ex.SEEDS)
p.add_argument("--models", nargs="+", default=ex.BENCHMARK_MODELS)
p.add_argument("--skip-benchmark", action="store_true")
p.add_argument("--skip-protocols", action="store_true")
p.add_argument("--ph2-root", default=None)
a = p.parse_args()

cache, results, analysis = (os.path.join(a.project, d) for d in ("cache", "results", "analysis"))
data, eval_sets = ex.prepare_data(a.raw, cache, a.size, external=("isic2017_test",), ph2_root=a.ph2_root)
if not a.skip_benchmark:
    ex.run_benchmark(data, eval_sets, results, models=a.models, seeds=a.seeds)
if not a.skip_protocols:
    ex.run_protocols(data, eval_sets, results, seeds=a.seeds)
print(open(ex.analyse(results, analysis, data, eval_sets)).read())
