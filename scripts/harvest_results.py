#!/usr/bin/env python3
"""harvest_results.py — turn raw run outputs into the paper's tables."""
from __future__ import annotations
import argparse, json, re, statistics as st
from pathlib import Path
from collections import defaultdict

# optional scipy for Wilcoxon; degrade gracefully
try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# Held-out class names used to split per_class_f1 into seen/unseen for baselines.
UNSEEN_CLASSES: set[str] = {"elevator_up", "elevator_down", "walk_upstairs", "walk_downstairs"}


def _get(d: dict, *keys):
    """First present key among aliases."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def classify_method(path: Path) -> str:
    s = str(path).lower()
    if "continuous" in s:
        return "continuous (SensorLLM-style)"
    if "gpt2" in s:
        return "ours (discrete)"
    # baselines/<model>/...
    parts = path.parts
    if "baselines" in parts:
        i = parts.index("baselines")
        if i + 1 < len(parts):
            return parts[i + 1]
    return path.parent.name


def parse_meta(path: Path) -> dict:
    """Pull split / k / rank / seed from a dir or file name."""
    name = str(path)
    meta = {}
    for key, pat in [("split", r"split([A-Za-z0-9]+)"), ("k", r"_k(\d+)"),
                     ("rank", r"_r(\d+)"), ("seed", r"_s(\d+)")]:
        m = re.search(pat, name)
        if m:
            meta[key] = m.group(1)
    return meta


def load_record(jpath: Path) -> dict | None:
    try:
        d = json.loads(jpath.read_text())
    except Exception:
        return None
    seen = _get(d, "seen_macro_f1", "test_seen_f1")
    unseen = _get(d, "unseen_macro_f1", "test_unseen_f1")
    overall = _get(d, "overall_macro_f1", "test_macro_f1", "macro_f1")
    # Baselines (protonet/attend/deepconvlstm/chronos/simple_finetune) emit only.
    if (seen is None or unseen is None) and isinstance(d.get("per_class_f1"), dict) and UNSEEN_CLASSES:
        pcf = d["per_class_f1"]
        un = [v for c, v in pcf.items() if c in UNSEEN_CLASSES]
        sn = [v for c, v in pcf.items() if c not in UNSEEN_CLASSES]
        if unseen is None and un:
            unseen = sum(un) / len(un)
        if seen is None and sn:
            seen = sum(sn) / len(sn)
    if overall is None and seen is None and unseen is None:
        return None  # not a results file
    meta = parse_meta(jpath.parent if jpath.parent != Path() else jpath)
    meta.update(parse_meta(jpath))
    return {
        "method": classify_method(jpath),
        "split": meta.get("split", _get(d, "split") or "?"),
        "k": meta.get("k", str(_get(d, "k_shot") or "?")),
        "seed": meta.get("seed", str(_get(d, "seed") or "?")),
        "seen": seen, "unseen": unseen, "overall": overall,
        "path": str(jpath),
    }


def discover(root: Path) -> list[dict]:
    recs = {}
    for jpath in root.rglob("*.json"):
        if jpath.name in {"adapter_config.json", "tokenizer_config.json",
                          "tokenizer.json", "config.json", "special_tokens_map.json",
                          "generation_config.json"}:
            continue
        r = load_record(jpath)
        if not r:
            continue
        # prefer eval.json over summary.json when both exist in same dir
        key = (r["method"], r["split"], r["k"], r["seed"], str(jpath.parent))
        if key in recs and jpath.name == "summary.json":
            continue
        recs[key] = r
    return list(recs.values())


def fmt(vals: list[float]) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{st.mean(vals):.3f} ± {st.pstdev(vals):.3f}"


def aggregate(recs: list[dict]):
    """group by (method, split, k) -> dict(metric -> list over seeds)."""
    g = defaultdict(lambda: {"seen": [], "unseen": [], "overall": [], "seeds": {}})
    for r in recs:
        key = (r["method"], r["split"], r["k"])
        for m in ("seen", "unseen", "overall"):
            if r[m] is not None:
                g[key][m].append(float(r[m]))
        g[key]["seeds"][r["seed"]] = r  # for paired tests
    return g


def md_main_table(g, split="C") -> str:
    rows = sorted({k for k in g if k[1] == split}, key=lambda k: k[0])
    out = [f"### Main results — Split {split} (mean ± std over seeds)\n",
           "| Method | K | Seen F1 | Unseen F1 | Overall F1 |",
           "|---|---|---|---|---|"]
    for (method, sp, k) in rows:
        cell = g[(method, sp, k)]
        out.append(f"| {method} | {k} | {fmt(cell['seen'])} | "
                   f"**{fmt(cell['unseen'])}** | {fmt(cell['overall'])} |")
    return "\n".join(out) + "\n"


def md_headline(g) -> str:
    """Discrete vs continuous on unseen, per split, with Wilcoxon over seeds."""
    out = ["### HEADLINE — discrete vs continuous (Unseen F1)\n",
           "| Split | Continuous | Discrete (ours) | Δ | Wilcoxon p |",
           "|---|---|---|---|---|"]
    splits = sorted({k[1] for k in g})
    for sp in splits:
        # pick the matching k (prefer largest k available for both)
        def pick(method):
            cands = {k[2]: g[k] for k in g if k[0] == method and k[1] == sp}
            return cands
        cont = pick("continuous (SensorLLM-style)")
        ours = pick("ours (discrete)")
        ks = sorted(set(cont) & set(ours), key=lambda x: (x != "?", x))
        if not ks:
            out.append(f"| {sp} | {fmt(sum((cont[k]['unseen'] for k in cont), []))} "
                       f"| {fmt(sum((ours[k]['unseen'] for k in ours), []))} | -- | -- |")
            continue
        k = ks[-1]
        cu, ou = cont[k]["unseen"], ours[k]["unseen"]
        delta = (st.mean(ou) - st.mean(cu)) if cu and ou else None
        # paired Wilcoxon by seed
        p = "--"
        cseeds, oseeds = cont[k]["seeds"], ours[k]["seeds"]
        common = sorted(set(cseeds) & set(oseeds))
        if _HAVE_SCIPY and len(common) >= 5:
            a = [float(cseeds[s]["unseen"]) for s in common if cseeds[s]["unseen"] is not None]
            b = [float(oseeds[s]["unseen"]) for s in common if oseeds[s]["unseen"] is not None]
            if len(a) == len(b) and len(a) >= 5 and any(x != y for x, y in zip(a, b)):
                try:
                    p = f"{wilcoxon(b, a).pvalue:.3f}"
                except Exception:
                    p = "n/a"
        elif not _HAVE_SCIPY:
            p = "install scipy"
        dtxt = f"{delta:+.3f}" if delta is not None else "--"
        out.append(f"| {sp} | {fmt(cu)} | **{fmt(ou)}** | {dtxt} | {p} |")
    return "\n".join(out) + "\n"


def latex_main(g, split="C") -> str:
    rows = sorted({k for k in g if k[1] == split}, key=lambda k: k[0])
    lines = [r"\begin{table}[t]\centering",
             rf"\caption{{Main results on Split {split} (mean$\pm$std).}}",
             r"\begin{tabular}{lcccc}", r"\hline",
             r"Method & K & Seen & Unseen & Overall \\ \hline"]
    for (method, sp, k) in rows:
        c = g[(method, sp, k)]
        def L(v):
            v = [x for x in v if x is not None]
            if not v: return "--"
            return f"{st.mean(v):.3f}$\\pm${st.pstdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"
        lines.append(f"{method} & {k} & {L(c['seen'])} & {L(c['unseen'])} & {L(c['overall'])} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="checkpoints", help="dir to scan for result JSONs")
    ap.add_argument("--out", default="results_tables", help="output dir for tables")
    ap.add_argument("--split", default="C", help="headline split for the main table")
    ap.add_argument("--unseen", default="elevator_up,elevator_down,walk_upstairs,walk_downstairs",
                    help="held-out class names, used to derive seen/unseen F1 from per_class_f1 for baselines")
    args = ap.parse_args()

    global UNSEEN_CLASSES
    UNSEEN_CLASSES = {c.strip() for c in args.unseen.split(",") if c.strip()}

    root = Path(args.root)
    recs = discover(root)
    print(f"[harvest] scanned {root} — found {len(recs)} result records")
    if not recs:
        print("[harvest] no result files yet. Run the pipeline, then re-run this.")
        return
    g = aggregate(recs)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    md = (f"# HAR-ICL results (auto-generated)\n\n"
          f"Records: {len(recs)} · scipy: {_HAVE_SCIPY}\n\n"
          + md_headline(g) + "\n" + md_main_table(g, args.split))
    (out / "tables.md").write_text(md)
    (out / "table_main.tex").write_text(latex_main(g, args.split))
    print(md)
    print(f"[harvest] wrote {out/'tables.md'} and {out/'table_main.tex'}")
    if not _HAVE_SCIPY:
        print("[harvest] tip: pip install scipy  -> enables Wilcoxon p-values")


if __name__ == "__main__":
    main()
