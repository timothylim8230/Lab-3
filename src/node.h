/*
 * node.h - EV charging node (Task 1)
 * FIT3143 Lab 3
 */
#ifndef NODE_H
#define NODE_H

#include "common.h"

/*
 * Runs one EV charging node.
 *   cfg       : runtime configuration
 *   node_comm : communicator containing only the m*n node ranks (from MPI_Comm_split)
 *   world     : MPI_COMM_WORLD (used for node <-> base station traffic)
 *   base_rank : rank of the base station in `world`
 * Returns when the termination message from the base station has been processed.
 */
int node_main(const Config *cfg, MPI_Comm node_comm, MPI_Comm world, int base_rank);

#endif /* NODE_H */
