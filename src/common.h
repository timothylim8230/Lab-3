/*
 * common.h - shared definitions for the EV charging WSN simulation
 * FIT3143 Lab 3
 *
 * Message structs, MPI tags, runtime configuration and helpers used by the
 * EV charging nodes (node.c). The base station message types (Report,
 * BaseReply, TAG_REPORT, TAG_BASE_REPLY, TAG_TERMINATE) are defined here
 * because the nodes send/receive them; the base station itself (Task 2) is
 * not part of this code base.
 */
#ifndef COMMON_H
#define COMMON_H

#include <mpi.h>
#include <stddef.h>
#include <stdio.h>

/* Communication version: 1 = blocking, 2 = non-blocking (default), 3 = non-blocking + persistent requests */
#ifndef COMM_VERSION
#define COMM_VERSION 2
#endif

#define MAX_PORTS   16   /* fixed-size shared array per node (spec allows a fixed size) */
#define MAX_NBRS    4    /* 2D Cartesian grid: top, bottom, left, right */
#define MAX_SUGGEST 8    /* max nearest available nodes returned by the base station */
#define HISTORY_LEN 64   /* ring buffer of per-cycle availability records (Table 1 style) */

/* MPI message tags. Each tag is one "message type" for the Task 3 analysis. */
enum {
    TAG_NBR_REQ    = 10, /* node -> neighbour : request availability           (cart_comm)  */
    TAG_NBR_REPLY  = 11, /* neighbour -> node : availability reply             (cart_comm)  */
    TAG_REPORT     = 20, /* node -> base      : congestion incident report     (world)      */
    TAG_BASE_REPLY = 21, /* base -> node      : nearest available nodes        (world)      */
    TAG_TERMINATE  = 30  /* base -> node      : graceful shutdown              (world)      */
};

/* Neighbour index convention used everywhere */
enum { NBR_TOP = 0, NBR_BOTTOM = 1, NBR_LEFT = 2, NBR_RIGHT = 3 };
extern const char *const NBR_NAME[MAX_NBRS];

/* Runtime configuration parsed from the command line (identical for every rank) */
typedef struct {
    int    m, n;             /* grid rows, columns                                   */
    int    k;                /* charging ports per node                              */
    int    threshold;        /* availability (free ports) below this => congested    */
    double cycle_s;          /* port status update cycle in seconds                  */
    double period_s;         /* base station "predefined period" T in seconds        */
    double duration_s;       /* simulation length in seconds (base station decides)  */
    unsigned seed;           /* RNG seed                                             */
    double congestion_prob;  /* probability a port becomes in-use at each update     */
    char   log_dir[256];     /* where logs are written                               */
} Config;

/* ---- Message payloads -------------------------------------------------
 * Layout rule: all int members first, then all double members. The MPI
 * derived datatypes in common.c rely on this (one MPI_INT block followed by
 * one MPI_DOUBLE block), so keep it when adding fields.
 */

/* node -> neighbour */
typedef struct {
    int    src_rank;      /* cart rank of the requesting node */
    int    seq;           /* incident sequence number (echoed in the reply) */
    double sent_at;       /* MPI_Wtime at send */
} NbrRequest;

/* neighbour -> node */
typedef struct {
    int    src_rank;      /* cart rank of the replying neighbour */
    int    seq;           /* echoed from the request */
    int    avail;         /* free ports at the neighbour */
    int    k;             /* number of ports at the neighbour */
    int    ports[MAX_PORTS];
    double req_sent_at;   /* echoed from the request -> RTT at the requester */
    double sent_at;       /* MPI_Wtime at send -> one-way delay at the requester */
} NbrReply;

/* node -> base station: congestion incident report */
typedef struct {
    int    seq;
    int    rank, cart_rank;
    int    coord[2];
    int    ts[6];                     /* year, month, day, hour, min, sec of discovery */
    int    k, avail, threshold;
    int    ports[MAX_PORTS];          /* raw port snapshot: 1 = in use, 0 = free */
    int    n_nbrs;                    /* number of existing neighbours */
    int    nbr_rank[MAX_NBRS];        /* MPI_PROC_NULL (-ish, we store -1) if absent */
    int    nbr_coord[MAX_NBRS][2];
    int    nbr_avail[MAX_NBRS];       /* -1 if the neighbour did not reply */
    int    nbr_replied[MAX_NBRS];
    int    inc_msgs_sent, inc_msgs_recv, inc_bytes_sent, inc_bytes_recv;     /* this incident */
    int    tot_msgs_sent, tot_msgs_recv, tot_bytes_sent, tot_bytes_recv;     /* cumulative with neighbours */
    double nbr_rtt[MAX_NBRS];         /* request->reply round trip per neighbour (s), -1 if none */
    double exchange_time;             /* total time spent collecting neighbour data (s) */
    double sent_at;                   /* MPI_Wtime at send */
} Report;

/* base station -> node */
typedef struct {
    int    seq;                       /* echoed report seq */
    int    n_available;               /* 0 => "no nodes available nearby" */
    int    ranks[MAX_SUGGEST];
    int    coords[MAX_SUGGEST][2];
    double report_sent_at;            /* echoed -> RTT at the node */
    double sent_at;
} BaseReply;

/* base -> node termination carries only the send time (double) for delay measurement */

typedef struct {
    MPI_Datatype nbr_req, nbr_reply, report, base_reply;
} MpiTypes;

void mpi_types_create(MpiTypes *t);
void mpi_types_free(MpiTypes *t);

/* Wall-clock timestamp helpers */
void wall_timestamp(int out[6]);
void format_timestamp(const int ts[6], char *buf, size_t n);

/* Sleep without burning CPU */
void sleep_ms(double ms);

/* mkdir -p style helper; returns 0 on success or if it already exists */
int ensure_dir(const char *path);

/* Print the config block to a log file (used by node and base logs) */
void config_dump(const Config *cfg, FILE *f);

#endif /* COMMON_H */
