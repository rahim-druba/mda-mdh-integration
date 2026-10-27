#!/bin/bash
# Full block-size x grid-size sweep, matching the granularity of the MDA
# technology table (N x Block configuration), for all three kernels:
# Integrated (bridge), MDH-alone (hand-written spec), MDA-alone (naive).
#
# Serial baseline is computed once per N (cached), not once per block config,
# since it does not depend on which GPU kernel/block shape is being compared
# (serial CPU time grows O(N^4), so re-running it per config would be very slow).
set -e
cd "$(dirname "$0")/.."   # repo root

BIN=/tmp/sweep_bin
mkdir -p "$BIN"
OUT="bench/sweep_results.csv"
echo "label,N,block,iterations,gpu_s,cpu_s,speedup,u_center_gpu,u_center_serial,gpu_exact_err" > "$OUT"

GRID_SIZES="64 128 256 512 1024"
# A=NUM_WI_L_1 (rows/y)  B=NUM_WI_L_2 (cols/x) -- matches MDA table's "AxB" block label
BLOCKS="32 32
16 32
16 16
16 8"

echo "=== Step 1: serial baseline cache (once per N) ==="
for N in $GRID_SIZES; do
  CACHE="bench/serial_cache_NX${N}.txt"
  if [ -f "$CACHE" ]; then
    echo "[cache hit] N=$N -> $(cat "$CACHE")"
    continue
  fi
  echo "[serial] building and running N=$N ..."
  # no -O flag: matches nvcc's default (unoptimized) host-code compilation,
  # which is what every already-published serial timing in this repo used
  g++ -DNX_GRID=${N} bench/serial_cache_gen.cc -o "$BIN/serial_gen_${N}"
  SECONDS_START=$(date +%s)
  "$BIN/serial_gen_${N}"
  SECONDS_END=$(date +%s)
  echo "[serial] N=$N done in $((SECONDS_END-SECONDS_START)) s wall -> $(cat "$CACHE")"
done

echo "=== Step 2: GPU block-size sweep (3 kernels x 5 sizes x 4 blocks = 60 runs) ==="
RUN=0
TOTAL=60
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
      -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1
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
      -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1
    "$BIN/mdhalone_${N}_${A}x${B}" | grep '^CSVROW' | sed 's/^CSVROW,//' >> "$OUT"

    RUN=$((RUN+1))
    echo "[$RUN/$TOTAL] N=$N block=${A}x${B} kernel=mda"
    nvcc bench/bench_gpu_sweep_mda.cu bench/mda_kernel.cu \
      -o "$BIN/mda_${N}_${A}x${B}" -DNX_GRID=${N} -DN=${N} -DBLOCK_X=${B} -DBLOCK_Y=${A}
    "$BIN/mda_${N}_${A}x${B}" | grep '^CSVROW' | sed 's/^CSVROW,//' >> "$OUT"
  done
done <<< "$BLOCKS"

echo "=== DONE === results in $OUT"
wc -l "$OUT"
