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
 */
/**
 * Timothy Lim  33111472 tlim0034@student.monash.edu
 * Scott Nguyen 33879095 sngu0065@student.monash.edu
 */
#include "common.h"
#include "node.h"
#include "base.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/**
 * Outlines how to run the simulation, providing order of and meaning behind each parameter
 */
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
    // if not enough args passed then stop
    if (argc < 8) {
        if (rank == 0) usage(argv[0]);
        return -1;
    }

    // reads args and stores them in config structure
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

    // sets up logging directory 
    const char *dir = getenv("WSN_LOG_DIR");
    snprintf(cfg->log_dir, sizeof(cfg->log_dir), "%s", (dir && *dir) ? dir : "logs");

    // checks if args passed are valid
    if (cfg->m < 1 || cfg->n < 1) { if (rank == 0) fprintf(stderr, "ERROR: m and n must be >= 1\n"); return -1; }
    if (cfg->k < 1 || cfg->k > MAX_PORTS) { if (rank == 0) fprintf(stderr, "ERROR: k must be in 1..%d\n", MAX_PORTS); return -1; }
    if (cfg->threshold < 0 || cfg->threshold > cfg->k) { if (rank == 0) fprintf(stderr, "ERROR: threshold must be in 0..k\n"); return -1; }
    if (cfg->cycle_s <= 0 || cfg->period_s <= 0 || cfg->duration_s <= 0) { if (rank == 0) fprintf(stderr, "ERROR: cycle, period and duration must be > 0\n"); return -1; }
    if (cfg->congestion_prob < 0 || cfg->congestion_prob > 1) { if (rank == 0) fprintf(stderr, "ERROR: congestion_prob must be in [0,1]\n"); return -1; }
    return 0;
}

int main(int argc, char **argv)
{
    int provided = 0,  
    rank, // process rank
    size, //total MPI processes in comm world
    rc = 0;

    Config cfg; // config storage

    // Only the OpenMP master thread of each node calls MPI => FUNNELED is enough
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // checks if mpi implementation support. must be able to support funneled
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "ERROR: MPI library does not provide MPI_THREAD_FUNNELED (got %d)\n", provided);
        MPI_Finalize();
        return 1;
    }

    // parses args and if it fails then stop
    if (parse_args(argc, argv, &cfg, rank) != 0) {
        MPI_Finalize();
        return 1;
    }

    // calculating nodes and rank of base station
    const int n_nodes   = cfg.m * cfg.n;
    const int base_rank = n_nodes;            
    
    
    // checks if number of processes can accomodate nodes + base. if not, then stop
    if (size != n_nodes + 1) {
        if (rank == 0)
            fprintf(stderr, "ERROR: need exactly m*n+1 = %d MPI processes for a %d x %d grid, but got %d\n",
                    n_nodes + 1, cfg.m, cfg.n, size);
        MPI_Finalize();
        return 1;
    }


    // ensures logging directory exists so each rank can write to it
    ensure_dir(cfg.log_dir);  

    // split world into {nodes} and {base station} 
    MPI_Comm split_comm;
    const int colour = (rank == base_rank) ? 1 : 0; // 0 for charging nodes, 1 for base 
    MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &split_comm);

    
    if (rank == base_rank) {
        rc = base_main(&cfg, MPI_COMM_WORLD, n_nodes);
    } else {
        rc = node_main(&cfg, split_comm, MPI_COMM_WORLD, base_rank);
    }

    MPI_Comm_free(&split_comm);
    MPI_Barrier(MPI_COMM_WORLD);   /* everyone has finished its own graceful shutdown */
    if (rank == base_rank) {
        printf("[base] all processes reached the final barrier, finalising MPI\n");
        fflush(stdout);
    }

    MPI_Finalize();
    return rc;
}
