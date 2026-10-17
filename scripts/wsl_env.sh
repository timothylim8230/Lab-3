#!/bin/bash
# Source this in WSL (Ubuntu) before building or running:
#   source scripts/wsl_env.sh
# Open MPI 5.0.8 was built from source (no sudo available) into ~/opt/openmpi.
export OMPI_HOME="$HOME/opt/openmpi"
export PATH="$OMPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OMPI_HOME/lib:${LD_LIBRARY_PATH:-}"
# Allow more processes than cores on a laptop (needed for larger grids in Task 3)
export OMPI_MCA_rmaps_base_oversubscribe=1
# Keep OpenMP threads from spinning after a parallel region (frees cores for the other ranks)
export OMP_WAIT_POLICY=passive
