"""Iterate configs/ablations/*.yaml, submit SLURM jobs, aggregate results."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from itertools import product
from pathlib import Path

import yaml

from .stats import aggregate_seeds


def load_ablation(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def enumerate_runs(ablation: dict) -> list[dict]:
    vqvae_cfgs = ablation.get("vqvae_configs", [ablation.get("vqvae_config")])
    gpt2_cfgs = ablation.get("gpt2_configs", [ablation.get("gpt2_config")])
    splits = ablation.get("splits", ["C"])
    seeds = ablation.get("seeds", [42])
    out = []
    for vq, g, sp, sd in product(filter(None, vqvae_cfgs), filter(None, gpt2_cfgs), splits, seeds):
        out.append({
            "ablation_id": ablation.get("id"),
            "ablation_name": ablation.get("name"),
            "vqvae_cfg": vq, "gpt2_cfg": g, "split": sp, "seed": sd,
            "baseline": ablation.get("baseline"),
        })
    return out


def submit_run(run: dict, scripts: dict[str, str]) -> dict:
    """Submit a single run via sbatch."""
    if run.get("baseline"):
        # Baseline-only ablation (e.g.
        env = {
            "BASELINE": run["baseline"],
            "SPLIT": run["split"],
            "SEED": str(run["seed"]),
        }
        cmd = ["sbatch", scripts["baseline"]]
    else:
        env = {
            "EXP_NAME": run["vqvae_cfg"],
            "GPT_CFG": run["gpt2_cfg"],
            "SPLIT": run["split"],
            "SEED": str(run["seed"]),
        }
        cmd = ["sbatch", scripts["main"]]
    print(f"[submit] {run} env={env}")
    full_env = {**dict(yaml.safe_load(yaml.safe_dump(env)).items())}
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env={**__import__("os").environ, **{k: str(v) for k, v in full_env.items()}}
    )
    return {"run": run, "stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}


def collect_results(out_root: Path) -> dict:
    """Glob summary.json files under out_root and aggregate per-config metrics."""
    by_key: dict[str, list[str]] = {}
    for p in out_root.rglob("summary.json"):
        # Path schema: checkpoints/gpt2/split{X}_k{K}_r{R}_s{S}/summary.json
        key = p.parent.parent.name + "/" + p.parent.name.rsplit("_s", 1)[0]
        by_key.setdefault(key, []).append(str(p))
    table = {}
    for k, files in by_key.items():
        table[k] = aggregate_seeds(files)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, required=True, help="dir of ablation YAMLs")
    parser.add_argument("--mode", choices=["submit", "collect", "enumerate"], default="enumerate")
    parser.add_argument("--main_sbatch", type=str, default="slurm/phase3_gpt2_array.sbatch")
    parser.add_argument("--baseline_sbatch", type=str, default="slurm/phase5_baselines.sbatch")
    parser.add_argument("--out_root", type=str, default="checkpoints/gpt2")
    parser.add_argument("--data", type=str, default="data/processed/v1")
    args = parser.parse_args()

    cfg_dir = Path(args.configs)
    ablations = [load_ablation(p) for p in sorted(cfg_dir.glob("ablation_*.yaml"))]
    runs = [r for ab in ablations for r in enumerate_runs(ab)]
    if args.mode == "enumerate":
        print(json.dumps(runs, indent=2))
        return
    if args.mode == "submit":
        scripts = {"main": args.main_sbatch, "baseline": args.baseline_sbatch}
        results = [submit_run(r, scripts) for r in runs]
        print(json.dumps(results, indent=2))
        return
    if args.mode == "collect":
        table = collect_results(Path(args.out_root))
        print(json.dumps(table, indent=2))
        return


if __name__ == "__main__":
    main()
