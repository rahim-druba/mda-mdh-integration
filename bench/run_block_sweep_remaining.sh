#!/bin/bash
# Fills in only the 9 rows missing from bench/sweep_results.csv: block 16x32,
# 16x16, and 16x8, each at N=1024 (3 kernels each). Appends to the existing
# CSV instead of overwriting it, since the other 51 rows are already valid.
# Serial baseline for N=1024 is already cached in bench/serial_cache_NX1024.txt.
set -e
cd "$(dirname "$0")/.."   # repo root

BIN=/tmp/sweep_bin
mkdir -p "$BIN"
OUT="bench/sweep_results.csv"
N=1024
NI=$((N-2))

BLOCKS="16 32
16 16
16 8"

RUN=0
TOTAL=9
while read -r A B; do
  [ -z "$A" ] && continue
  NWG1=$((N/A))
  NWG2=$((N/B))

  RUN=$((RUN+1))
  echo "[$RUN/$TOTAL] N=$N block=${A}x${B} kernel=integrated"
  nvcc bench/bench_gpu_sweep_mdh.cu generated/poisson_1.cu \
    -o "$BIN/int_${N}_${A}x${B}" -DNX_GRID=${N} -DSWEEP_LABEL='"integrated"' \
    -DTYPE_T=double -DTYPE_TS=double -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
    -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=${NI} -DG_CB_SIZE_L_2=${NI} \
    -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=${A} -DL_CB_SIZE_L_2=${B} \
    -DNUM_WG_L_1=${NWG1} -DNUM_WG_L_2=${NWG2} -DNUM_WI_L_1=${A} -DNUM_WI_L_2=${B} \
    -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0 \
    -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1
  "$BIN/int_${N}_${A}x${B}" | tee -a bench/sweep_remaining_log.txt | grep '^CSVROW' | sed 's/^CSVROW,//' >> "$OUT"

  RUN=$((RUN+1))
  echo "[$RUN/$TOTAL] N=$N block=${A}x${B} kernel=mdh_alone"
  nvcc bench/bench_gpu_sweep_mdh.cu generated/handwritten/poisson_1.cu \
    -o "$BIN/mdhalone_${N}_${A}x${B}" -DNX_GRID=${N} -DSWEEP_LABEL='"mdh_alone"' \
    -DTYPE_T=double -DTYPE_TS=double -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
    -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=${NI} -DG_CB_SIZE_L_2=${NI} \
    -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=${A} -DL_CB_SIZE_L_2=${B} \
    -DNUM_WG_L_1=${NWG1} -DNUM_WG_L_2=${NWG2} -DNUM_WI_L_1=${A} -DNUM_WI_L_2=${B} \
    -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0 \
    -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1
  "$BIN/mdhalone_${N}_${A}x${B}" | tee -a bench/sweep_remaining_log.txt | grep '^CSVROW' | sed 's/^CSVROW,//' >> "$OUT"

  RUN=$((RUN+1))
  echo "[$RUN/$TOTAL] N=$N block=${A}x${B} kernel=mda"
  nvcc bench/bench_gpu_sweep_mda.cu bench/mda_kernel.cu \
    -o "$BIN/mda_${N}_${A}x${B}" -DNX_GRID=${N} -DN=${N} -DBLOCK_X=${B} -DBLOCK_Y=${A}
  "$BIN/mda_${N}_${A}x${B}" | tee -a bench/sweep_remaining_log.txt | grep '^CSVROW' | sed 's/^CSVROW,//' >> "$OUT"
done <<< "$BLOCKS"

echo "=== DONE === appended 9 rows to $OUT"
wc -l "$OUT"
