#!/bin/bash
# GPU block-size sweep for N=64/128/256/512 only, run in parallel with the
# still-running N=1024 serial baseline (bench/run_block_sweep.sh). Uses the
# serial caches already written for 64/128/256/512. Safe to run concurrently:
# this is GPU-bound, the N=1024 serial job is CPU-bound.
set -e
cd "$(dirname "$0")/.."

BIN=/tmp/sweep_bin
mkdir -p "$BIN"
OUT="bench/sweep_results_partial.csv"
echo "label,N,block,iterations,gpu_s,cpu_s,speedup,u_center_gpu,u_center_serial,gpu_exact_err" > "$OUT"

GRID_SIZES="64 128 256 512"
BLOCKS="32 32
16 32
16 16
16 8"

RUN=0
TOTAL=48
while read -r A B; do
  [ -z "$A" ] && continue
  for N in $GRID_SIZES; do
    NI=$((N-2))
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
      -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1 2>/dev/null
    "$BIN/int_${N}_${A}x${B}" | grep '^CSVROW' | sed 's/^CSVROW,//' >> "$OUT"

    RUN=$((RUN+1))
    echo "[$RUN/$TOTAL] N=$N block=${A}x${B} kernel=mdh_alone"
    nvcc bench/bench_gpu_sweep_mdh.cu generated/handwritten/poisson_1.cu \
      -o "$BIN/mdhalone_${N}_${A}x${B}" -DNX_GRID=${N} -DSWEEP_LABEL='"mdh_alone"' \
      -DTYPE_T=double -DTYPE_TS=double -DCACHE_L_CB=0 -DCACHE_P_CB=0 \
      -DG_CB_RES_DEST_LEVEL=2 -DG_CB_SIZE_L_1=${NI} -DG_CB_SIZE_L_2=${NI} \
      -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=${A} -DL_CB_SIZE_L_2=${B} \
      -DNUM_WG_L_1=${NWG1} -DNUM_WG_L_2=${NWG2} -DNUM_WI_L_1=${A} -DNUM_WI_L_2=${B} \
      -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0 \
      -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1 2>/dev/null
    "$BIN/mdhalone_${N}_${A}x${B}" | grep '^CSVROW' | sed 's/^CSVROW,//' >> "$OUT"

    RUN=$((RUN+1))
    echo "[$RUN/$TOTAL] N=$N block=${A}x${B} kernel=mda"
    nvcc bench/bench_gpu_sweep_mda.cu bench/mda_kernel.cu \
      -o "$BIN/mda_${N}_${A}x${B}" -DNX_GRID=${N} -DN=${N} -DBLOCK_X=${B} -DBLOCK_Y=${A} 2>/dev/null
    "$BIN/mda_${N}_${A}x${B}" | grep '^CSVROW' | sed 's/^CSVROW,//' >> "$OUT"
  done
done <<< "$BLOCKS"

echo "=== PARTIAL SWEEP DONE === $OUT"
wc -l "$OUT"
