#!/bin/bash
# caas_common.sh - sourced by every scripts/caas_*.job on the CAAS cluster
# FIT3143 Lab 3
#
# Loads the Open MPI module, moves to the repository root (the directory the
# job was submitted from), builds the binaries with the cluster compiler and
# prints the allocation so the slurm-*.out file documents where the run went.

module load openmpi/4.1.5-gcc-11.2.0-ux65npg

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1

echo "=== CAAS job ${SLURM_JOB_ID:-?} (${SLURM_JOB_NAME:-?}) on $(hostname) at $(date -Is)"
echo "=== nodes: ${SLURM_JOB_NODELIST:-?}  ntasks=${SLURM_NTASKS:-?}  per-node=${SLURM_NTASKS_PER_NODE:-?}  cpus/task=${SLURM_CPUS_PER_TASK:-?}"
mpicc --version | head -1
mpirun --version | head -1

# Scripts edited on Windows may carry CRLF endings; the cluster's bash rejects them.
command -v dos2unix >/dev/null 2>&1 && dos2unix -q scripts/*.sh 2>/dev/null

# Rebuild only if sources changed (make handles that). Fail loudly if it breaks.
make all || { echo "build failed"; exit 1; }

export OMP_WAIT_POLICY=passive
mkdir -p results
