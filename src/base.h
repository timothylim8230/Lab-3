/* base.h - Task 2 base station */
#ifndef BASE_H
#define BASE_H

#include "common.h"

/* Called only by the dedicated base-station MPI rank. */
int base_main(const Config *cfg, MPI_Comm world, int n_nodes);

#endif
