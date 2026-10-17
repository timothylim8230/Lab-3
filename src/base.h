/* base.h - Task 2 base station */
/**
 * Timothy Lim  33111472 tlim0034@student.monash.edu
 * Scott Nguyen 33879095 sngu0065@student.monash.edu
 */
#ifndef BASE_H
#define BASE_H

#include "common.h"

/* Called only by the dedicated base-station MPI rank. */
int base_main(const Config *cfg, MPI_Comm world, int n_nodes);

#endif
