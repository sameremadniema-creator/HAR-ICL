#!/bin/bash
# List which cluster files the paper needs (KEEP) vs which are superseded (DELETE.

set -uo pipefail
ROOT=${HAR_ROOT:-/mnt/fast/nobackup/users/aa08742}
REPO=${HAR_REPO:-$ROOT/har_icl}
sz(){ [ -e "$1" ] && du -sh "$1" 2>/dev/null | cut -f1 || echo "-"; }

echo "############################################################"
echo "#  KEEP — everything the paper's numbers trace to"
echo "############################################################"
echo "[final discrete readers]  (job 2166197 headline K=4 + job 2166198 K=1/2/8)"
ls -d "$ROOT"/checkpoints/gpt2/vqvae_usc_t32_splitC_k*_r16_s*_matched 2>/dev/null
echo "[final continuous control] (job 2166197)"
ls "$ROOT"/checkpoints/baselines/continuous_projection/splitC_k4_s*_matched.json 2>/dev/null
echo "[baselines] protonet/simple_finetune/attend/deepconvlstm(+attn)/chronos_mlp"
ls -d "$ROOT"/checkpoints/baselines/{protonet,simple_finetune,attend,deepconvlstm,deepconvlstm_attn,chronos_mlp} 2>/dev/null
echo "[sweep + confound JSONs] (jobs 2165037 / 2165035)"
ls "$ROOT"/tokens/*_s42/codebook_recovery_splitC.json 2>/dev/null | wc -l | xargs echo "  sweep recovery files:"
ls "$ROOT"/tokens/vqvae_aux_contrast_t32_s42/recovery_*.json 2>/dev/null
echo "[reproducibility upstream — DO NOT DELETE] encoders / tokens / processed data"
echo "  vqvae ckpts: $(sz "$ROOT"/checkpoints/vqvae)   tokens: $(sz "$ROOT"/tokens)   data: $(sz "$REPO"/data/processed)"
echo "[final logs] matched_2166197_* kshotm_2166198_* cbsweep_2165037* confound_2165035* clean_2163544_*"

echo
echo "############################################################"
echo "#  DELETE CANDIDATES — superseded, not used in the paper"
echo "############################################################"
# pre-fix discrete readers (no _matched suffix) and smoke runs
echo "-- pre-fix / smoke discrete GPT-2 checkpoints --"
for d in "$ROOT"/checkpoints/gpt2/*; do
  case "$d" in
    *_matched) : ;;                                   # keep final
    *vqvae_usc_t32_splitC_k*_r16_s*) echo "  $(sz "$d")  $d" ;;   # pre-fix reader
    *_smoke) echo "  $(sz "$d")  $d" ;;
  esac
done
echo "-- pre-fix / smoke continuous control JSONs --"
for f in "$ROOT"/checkpoints/baselines/continuous_projection/*.json; do
  case "$f" in *_matched.json) : ;; *) echo "  $(sz "$f")  $f" ;; esac
done
echo "-- superseded SLURM logs (old matched/kshot/cont/sweep/confound attempts) --"
for pat in matched_2165751 matched_2165818 matched_2165835 matched_2165837 kshotm_21658 \
           cont_2164 kshot_2164887 cbsweep_2165030 cbsweep_2165036 confound_2165029 matchedsmoke; do
  ls "$REPO"/logs/slurm/${pat}* 2>/dev/null
done

echo
echo "############################################################"
echo "#  READY-TO-PASTE removal commands (review, then run if you agree)"
echo "############################################################"
echo "# pre-fix + smoke discrete readers:"
echo "find $ROOT/checkpoints/gpt2 -maxdepth 1 -type d \\( -name 'vqvae_usc_t32_splitC_k*_r16_s*' ! -name '*_matched' -o -name '*_smoke' \\) -print"
echo "#   (swap -print for -exec rm -rf {} + once you've eyeballed the list)"
echo "# pre-fix + smoke continuous JSONs:"
echo "ls $ROOT/checkpoints/baselines/continuous_projection/*.json | grep -v _matched.json"
echo "# superseded logs:"
echo "ls $REPO/logs/slurm/{matched_2165751,matched_2165818,matched_2165835,matched_2165837,kshotm_21658,cont_2164,kshot_2164887,cbsweep_2165030,cbsweep_2165036,confound_2165029,matchedsmoke}* 2>/dev/null"
echo
echo "NOTE: keep checkpoints/vqvae, tokens/, and data/processed — deleting those would prevent re-running."
