#!/bin/bash
# Collect EVERY result that appears in the paper into one bundle, with provenance.

set -uo pipefail
ROOT=${HAR_ROOT:-/mnt/fast/nobackup/users/aa08742}
REPO=${HAR_REPO:-$ROOT/har_icl}
RANK=${RANK:-16}
SEEDS=(42 1337 2024 7 31337 99 2025 31415)
SWEEP=(vqvae_vanilla vqvae_aux_only vqvae_cond_only vqvae_aux_contrast vqvae_aux_t32 vqvae_aux_contrast_t32 \
       vqvae_fsq vqvae_fsq_t32 vqvae_fsq_contrast_t32 vqvae_full_k128 vqvae_full_k256 vqvae_full_k512 vqvae_full_k1024)
BASE_DIRS=(protonet simple_finetune attend deepconvlstm deepconvlstm_attn chronos_mlp)

OUT="$REPO/paper_evidence_bundle"
rm -rf "$OUT"; mkdir -p "$OUT"
MAN="$OUT/MANIFEST.tsv"
printf "role\tfile\tmtime\tsize_bytes\tsha256\n" > "$MAN"

# copy $1 (absolute) into the bundle under relative path from $ROOT, tag role $2.
add() {
  local f="$1" role="$2"
  if [ ! -f "$f" ]; then echo "  [MISSING] ($role) $f"; return 1; fi
  local rel="${f#$ROOT/}"; mkdir -p "$OUT/$(dirname "$rel")"; cp "$f" "$OUT/$rel"
  local mt sz sh
  mt=$(date -r "$f" '+%Y-%m-%d %H:%M:%S'); sz=$(stat -c%s "$f"); sh=$(sha256sum "$f" | cut -d' ' -f1)
  printf "%s\t%s\t%s\t%s\t%s\n" "$role" "$rel" "$mt" "$sz" "$sh" >> "$MAN"
  return 0
}

echo "=========================================================================="
echo "  COLLECTING PAPER EVIDENCE  (root: $ROOT)"
echo "=========================================================================="

# 1) headline discrete (K=4) + K-shot discrete (K=1,2,8)
for K in 4 1 2 8; do for s in "${SEEDS[@]}"; do
  add "$ROOT/checkpoints/gpt2/vqvae_usc_t32_splitC_k${K}_r${RANK}_s${s}_matched/eval.json" "discrete_k${K}"
done; done
# 2) matched continuous (K=4)
for s in "${SEEDS[@]}"; do
  add "$ROOT/checkpoints/baselines/continuous_projection/splitC_k4_s${s}_matched.json" "continuous"
done
# 3) baselines
for b in "${BASE_DIRS[@]}"; do for s in "${SEEDS[@]}"; do
  for f in "$ROOT/checkpoints/baselines/$b/splitC_k4_s${s}.json" "$ROOT/checkpoints/baselines/$b/splitC_s${s}.json"; do
    [ -f "$f" ] && add "$f" "baseline_${b}"
  done
done; done
# 4) tokeniser sweep (seed 42)
for c in "${SWEEP[@]}"; do add "$ROOT/tokens/${c}_s42/codebook_recovery_splitC.json" "sweep"; done
# 5) device confound (seed 42)
for c in crossdataset pamap2 usc; do add "$ROOT/tokens/vqvae_aux_contrast_t32_s42/recovery_${c}.json" "confound_${c}"; done
# 6) logs for the jobs above
for j in $(seq 0 7); do
  add "$REPO/logs/slurm/matched_2166197_${j}.out" "log_matched"
  add "$REPO/logs/slurm/kshotm_2166198_${j}.out" "log_kshot"
  add "$REPO/logs/slurm/clean_2163544_${j}.out" "log_clean_baselines"
done
add "$REPO/logs/slurm/cbsweep_2165037.out" "log_sweep"
add "$REPO/logs/slurm/confound_2165035.out" "log_confound"

echo
echo "=========================================================================="
echo "  VERIFICATION TABLE  (mtime = when the cluster wrote the file)"
echo "=========================================================================="
python3 - "$OUT" <<'PY'
import json, os, sys, glob, datetime
OUT=sys.argv[1]
def mt(p): return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%Y-%m-%d %H:%M')
def val(p):
    try: r=json.load(open(p))
    except Exception as e: return f"(unreadable: {e})"
    if "unseen_prototype_fewshot_macro_f1" in r:
        u=r["unseen_prototype_fewshot_macro_f1"].get("zq_mean"); n=r.get("reconstruction",{}).get("normalised_mse")
        extra=""
        if "name_that_dataset_control" in r: extra=f" name-that-dataset={r['name_that_dataset_control'].get('name_that_dataset_acc')}"
        return f"unseen={u} nMSE={n}{extra}"
    u=r.get("unseen_macro_f1", r.get("test_unseen_f1")); s=r.get("seen_macro_f1", r.get("test_seen_f1"))
    ch=r.get("chance",{}).get("unseen_macro_f1")
    return f"unseen={u} seen={s}" + (f" chance={ch}" if ch is not None else "")
rows=[]
for p in glob.glob(f"{OUT}/**/*.json", recursive=True):
    if p.endswith("MANIFEST.tsv"): continue
    rows.append((mt(p), os.path.relpath(p,OUT), val(p)))
for mtime,rel,v in sorted(rows):
    print(f"{mtime}   {rel:70s} {v}")
mts=sorted(set(r[0][:10] for r in rows))
print(f"\ndistinct file DATES present: {mts}")
print("→ all headline/k-shot/continuous files should share the newest date (the final rerun).")
print("→ any file with an OLDER date under a *_matched path is stale (a seed that did not rerun) — tell me and I'll flag it.")
PY

tar -czf "$REPO/paper_evidence_bundle.tgz" -C "$REPO" paper_evidence_bundle
echo
echo "bundle: $REPO/paper_evidence_bundle.tgz   ($(du -h "$REPO/paper_evidence_bundle.tgz" | cut -f1))"
echo "manifest: $OUT/MANIFEST.tsv"
echo "rsync it home with the command Claude gives you."
