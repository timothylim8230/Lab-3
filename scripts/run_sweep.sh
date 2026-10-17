#!/bin/bash
# run_sweep.sh - run the WSN simulation over many configurations (Task 3)
# FIT3143 Lab 3
#
# Usage:  scripts/run_sweep.sh <grid|line|period|versions|smoke> [out_dir]
#
# Works both on a laptop (mpirun --oversubscribe) and inside a SLURM job on
# CAAS (srun job steps, falling back to mpirun --oversubscribe for runs that
# need more ranks than the allocation has). Every run gets its own directory
# under out_dir with the node logs, base.log and the timing CSVs, plus one
# line in out_dir/index.csv so plot.py can find everything.
#
# Environment overrides (all optional):
#   WSN_BIN    binary to run                      (default ./wsn)
#   K          ports per node                     (default 6)
#   THRESHOLD  congested if free ports < this     (default 2)
#   CYCLE      port update cycle in seconds       (default 1)
#   PERIOD     base station period T in seconds   (default 10)
#   DURATION   seconds per run                    (default 8)
#   SEED       RNG seed                           (default 1)
#   PROB       congestion probability             (default 0.8)
#   PROB_PERIOD congestion probability for the period sweep (default 0.7)
#   REPEATS    repetitions per configuration      (default 1)
#   SIZES      "m,n m,n ..." list overriding the built-in grid/line lists
#   PERIODS    "T T ..." list for the period sweep (default "2 5 10 20 60")
#   LAUNCHER   auto | srun | mpirun               (default auto)
#   MAX_NP     ranks launchable with srun          (default $SLURM_NTASKS or 0)
set -euo pipefail

MODE=${1:?usage: $0 <grid|line|period|versions|smoke> [out_dir]}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${2:-results/${MODE}_${STAMP}${SLURM_JOB_ID:+_job$SLURM_JOB_ID}}

WSN_BIN=${WSN_BIN:-./wsn}
K=${K:-6}
THRESHOLD=${THRESHOLD:-2}
CYCLE=${CYCLE:-1}
PERIOD=${PERIOD:-10}
DURATION=${DURATION:-8}
SEED=${SEED:-1}
PROB=${PROB:-0.8}
REPEATS=${REPEATS:-1}
LAUNCHER=${LAUNCHER:-auto}
MAX_NP=${MAX_NP:-${SLURM_NTASKS:-0}}
PERIODS=${PERIODS:-"2 5 10 20 60"}

# >= 30 grid sizes (m,n) for Task 3(a); x-axis is m*n
GRID_SIZES="2,2 2,3 3,3 3,4 4,4 4,5 5,5 5,6 6,6 6,7 7,7 7,8 8,8 8,9 9,9 9,10 10,10 10,11 11,11 11,12 12,12 12,13 13,13 13,14 14,14 14,15 15,15 15,16 16,16 16,17 17,17"
# >= 30 line lengths 1xN for Task 3(b)
LINE_SIZES=$(for n in $(seq 2 33); do printf '1,%d ' "$n"; done)
# version comparison grids
VERSION_SIZES="2,2 3,3 4,4 5,5 6,6"

# OpenMP threads inside each rank sleep most of the time; never let them spin
export OMP_WAIT_POLICY=passive
export OMPI_MCA_rmaps_base_oversubscribe=1

mkdir -p "$OUT"
INDEX="$OUT/index.csv"
[ -f "$INDEX" ] || echo "mode,version,m,n,nodes,k,threshold,cycle_s,period_s,duration_s,seed,prob,rep,launcher,dir,exit_code,wall_s" > "$INDEX"

hostlist=${SLURM_JOB_NODELIST:-$(hostname)}
{
    echo "# sweep $MODE started $(date -Is)"
    echo "# host(s): $hostlist"
    echo "# slurm job: ${SLURM_JOB_ID:-none}  ntasks: ${SLURM_NTASKS:-n/a}  ntasks-per-node: ${SLURM_NTASKS_PER_NODE:-n/a}"
    echo "# binary: $WSN_BIN"
    command -v mpirun >/dev/null && mpirun --version 2>/dev/null | head -1
} > "$OUT/sweep_info.txt"

# launch <np> <args...>
launch() {
    local np=$1; shift
    local how=$LAUNCHER
    if [ "$how" = auto ]; then
        if [ -n "${SLURM_JOB_ID:-}" ] && [ "$np" -le "$MAX_NP" ]; then how=srun; else how=mpirun; fi
    fi
    case $how in
        srun)   srun --ntasks="$np" --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "$@" ;;
        mpirun) mpirun -np "$np" --oversubscribe "$@" ;;
        *)      echo "bad LAUNCHER=$how" >&2; return 2 ;;
    esac
    LAST_LAUNCHER=$how
}

# run_one <version_tag> <m> <n> <T> <rep> <prob>
run_one() {
    local ver=$1 m=$2 n=$3 T=$4 rep=$5 prob=$6
    local nodes=$((m * n)) np=$((m * n + 1))
    local dir="$OUT/${ver}_${m}x${n}_T${T}_r${rep}"
    mkdir -p "$dir"
    local t0 t1 rc=0
    t0=$(date +%s.%N)
    WSN_LOG_DIR="$dir" launch "$np" "$WSN_BIN" "$m" "$n" "$K" "$THRESHOLD" "$CYCLE" "$T" "$DURATION" "$((SEED + rep))" "$prob" \
        > "$dir/stdout.txt" 2> "$dir/stderr.txt" || rc=$?
    t1=$(date +%s.%N)
    local wall
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
    echo "$MODE,$ver,$m,$n,$nodes,$K,$THRESHOLD,$CYCLE,$T,$DURATION,$((SEED + rep)),$prob,$rep,$LAST_LAUNCHER,$dir,$rc,$wall" >> "$INDEX"
    printf '%-9s %-8s %3dx%-3d T=%-4s rep=%d  np=%-4d %-6s rc=%d  %6ss  alerts=%s\n' \
        "$MODE" "$ver" "$m" "$n" "$T" "$rep" "$np" "$LAST_LAUNCHER" "$rc" "$wall" \
        "$(grep -h 'alerts / reports received' "$dir/base.log" 2>/dev/null | awk '{print $NF}')"
    if [ "$rc" -ne 0 ]; then
        echo "  ** run failed; see $dir/stderr.txt" >&2
    fi
}

ver_tag() { basename "$WSN_BIN"; }

case $MODE in
    smoke)
        run_one "$(ver_tag)" 3 3 "$PERIOD" 0 "$PROB"
        ;;
    grid)
        for s in ${SIZES:-$GRID_SIZES}; do
            m=${s%,*}; n=${s#*,}
            for ((r = 0; r < REPEATS; r++)); do run_one "$(ver_tag)" "$m" "$n" "$PERIOD" "$r" "$PROB"; done
        done
        ;;
    line)
        for s in ${SIZES:-$LINE_SIZES}; do
            m=${s%,*}; n=${s#*,}
            for ((r = 0; r < REPEATS; r++)); do run_one "$(ver_tag)" "$m" "$n" "$PERIOD" "$r" "$PROB"; done
        done
        ;;
    period)
        # fixed grid, several values of T; moderate congestion so both reply types occur
        for s in ${SIZES:-4,4}; do
            m=${s%,*}; n=${s#*,}
            for T in $PERIODS; do
                for ((r = 0; r < REPEATS; r++)); do run_one "$(ver_tag)" "$m" "$n" "$T" "$r" "${PROB_PERIOD:-0.7}"; done
            done
        done
        ;;
    versions)
        for bin in ./wsn_v1 ./wsn ./wsn_v3; do
            [ -x "$bin" ] || { echo "skipping $bin (not built)"; continue; }
            WSN_BIN=$bin
            for s in ${SIZES:-$VERSION_SIZES}; do
                m=${s%,*}; n=${s#*,}
                for ((r = 0; r < REPEATS; r++)); do run_one "$(ver_tag)" "$m" "$n" "$PERIOD" "$r" "$PROB"; done
            done
        done
        ;;
    *)
        echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

echo "# sweep finished $(date -Is)" >> "$OUT/sweep_info.txt"
echo "done: $OUT  (index: $INDEX)"
