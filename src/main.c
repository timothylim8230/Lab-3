/*
 * main.c - entry point for the EV charging WSN simulation
 * FIT3143 Lab 3
 *
 * Process layout (P = m*n + 1 MPI processes):
 *   ranks 0 .. m*n-1 : EV charging nodes, arranged in an m x n Cartesian virtual topology
 *   rank  m*n        : base station (single MPI process, single thread)
 *
 * MPI_COMM_WORLD is split (MPI_Comm_split) into a node communicator and a
 * one-process base station communicator, following the master/slave pattern
 * from the Week 11 prep sheet. Node <-> base traffic uses MPI_COMM_WORLD.
 *
 * This code base contains Task 1 (EV charging nodes) only. The base station
 * (Task 2) is represented by base_placeholder() below, which exists solely so
 * the nodes can be run and shut down gracefully.
 */
#include "common.h"
#include "node.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Placeholder for the base station rank (NOT Task 2).
 *
 * Task 1 nodes keep running until they receive TAG_TERMINATE from the base
 * station rank, so some process has to occupy that rank and send the message.
 * This function waits for cfg->duration_s and then sends the termination
 * message to every node. It does not listen for congestion reports, compute
 * nearest available nodes, reply, or log anything; reports sent by the nodes
 * during the run go unanswered (the node logs a timeout warning and carries on).
 * Task 2 replaces this function with the real base station.
 */
static int base_placeholder(const Config *cfg, MPI_Comm world, int n_nodes)
{
    const double t0 = MPI_Wtime();
    printf("[base placeholder] no base station implemented; terminating %d nodes after %.1f s\n",
           n_nodes, cfg->duration_s);
    fflush(stdout);
    while (MPI_Wtime() - t0 < cfg->duration_s)
        sleep_ms(100);

    MPI_Request *req = malloc((size_t)n_nodes * sizeof *req);
    if (!req) {
        fprintf(stderr, "base placeholder: out of memory\n");
        return 1;
    }
    double t_term = MPI_Wtime();
    for (int i = 0; i < n_nodes; i++)
        MPI_Isend(&t_term, 1, MPI_DOUBLE, i, TAG_TERMINATE, world, &req[i]);
    MPI_Waitall(n_nodes, req, MPI_STATUSES_IGNORE);
    free(req);

    printf("[base placeholder] termination message sent to %d nodes\n", n_nodes);
    fflush(stdout);
    return 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: mpirun -np <m*n+1> %s <m> <n> <k> <threshold> <cycle_s> <period_T_s> <duration_s> [seed] [congestion_prob]\n"
        "  m, n            grid rows and columns (m*n EV charging nodes)\n"
        "  k               charging ports per node (1..%d)\n"
        "  threshold       node is congested when free ports < threshold\n"
        "  cycle_s         port status update cycle in seconds (e.g. 2)\n"
        "  period_T_s      base station predefined period T in seconds\n"
        "  duration_s      how long the base station runs before terminating all nodes\n"
        "  seed            optional RNG seed (default 1)\n"
        "  congestion_prob optional probability a port is in use after an update (default 0.6)\n"
        "Environment: WSN_LOG_DIR overrides the log directory (default ./logs)\n",
        prog, MAX_PORTS);
}

static int parse_args(int argc, char **argv, Config *cfg, int rank)
{
    if (argc < 8) {
        if (rank == 0) usage(argv[0]);
        return -1;
    }
    memset(cfg, 0, sizeof(*cfg));
    cfg->m               = atoi(argv[1]);
    cfg->n               = atoi(argv[2]);
    cfg->k               = atoi(argv[3]);
    cfg->threshold       = atoi(argv[4]);
    cfg->cycle_s         = atof(argv[5]);
    cfg->period_s        = atof(argv[6]);
    cfg->duration_s      = atof(argv[7]);
    cfg->seed            = (argc > 8) ? (unsigned)strtoul(argv[8], NULL, 10) : 1u;
    cfg->congestion_prob = (argc > 9) ? atof(argv[9]) : 0.6;

    const char *dir = getenv("WSN_LOG_DIR");
    snprintf(cfg->log_dir, sizeof(cfg->log_dir), "%s", (dir && *dir) ? dir : "logs");

    if (cfg->m < 1 || cfg->n < 1) { if (rank == 0) fprintf(stderr, "ERROR: m and n must be >= 1\n"); return -1; }
    if (cfg->k < 1 || cfg->k > MAX_PORTS) { if (rank == 0) fprintf(stderr, "ERROR: k must be in 1..%d\n", MAX_PORTS); return -1; }
    if (cfg->threshold < 0 || cfg->threshold > cfg->k) { if (rank == 0) fprintf(stderr, "ERROR: threshold must be in 0..k\n"); return -1; }
    if (cfg->cycle_s <= 0 || cfg->period_s <= 0 || cfg->duration_s <= 0) { if (rank == 0) fprintf(stderr, "ERROR: cycle, period and duration must be > 0\n"); return -1; }
    if (cfg->congestion_prob < 0 || cfg->congestion_prob > 1) { if (rank == 0) fprintf(stderr, "ERROR: congestion_prob must be in [0,1]\n"); return -1; }
    return 0;
}

int main(int argc, char **argv)
{
    int provided = 0, rank, size, rc = 0;
    Config cfg;

    /* Only the OpenMP master thread of each node calls MPI => FUNNELED is enough */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "ERROR: MPI library does not provide MPI_THREAD_FUNNELED (got %d)\n", provided);
        MPI_Finalize();
        return 1;
    }

    if (parse_args(argc, argv, &cfg, rank) != 0) {
        MPI_Finalize();
        return 1;
    }

    const int n_nodes   = cfg.m * cfg.n;
    const int base_rank = n_nodes;               /* last rank */
    if (size != n_nodes + 1) {
        if (rank == 0)
            fprintf(stderr, "ERROR: need exactly m*n+1 = %d MPI processes for a %d x %d grid, but got %d\n",
                    n_nodes + 1, cfg.m, cfg.n, size);
        MPI_Finalize();
        return 1;
    }

    ensure_dir(cfg.log_dir);   /* every rank may call this; EEXIST is fine */

    /* Split world into {nodes} and {base station} (Week 11 prep, Task 3 pattern) */
    MPI_Comm split_comm;
    const int colour = (rank == base_rank) ? 1 : 0;
    MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &split_comm);

    if (rank == base_rank) {
        rc = base_placeholder(&cfg, MPI_COMM_WORLD, n_nodes);
    } else {
        rc = node_main(&cfg, split_comm, MPI_COMM_WORLD, base_rank);
    }

    MPI_Comm_free(&split_comm);
    MPI_Barrier(MPI_COMM_WORLD);   /* everyone has finished its own graceful shutdown */
    if (rank == base_rank) {
        printf("[base placeholder] all processes reached the final barrier, finalising MPI\n");
        fflush(stdout);
    }
    MPI_Finalize();
    return rc;
}
