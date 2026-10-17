/*
 * base.c - FIT3143 Lab 3, Task 2: single-process base station
 *
 * Only this MPI process handles the base station; no OpenMP threads are used.
 * Rank assumptions match Task 1: reorder=0, so node world/cart ranks agree,
 * and a node at (row,col) has world rank row * n + col.
 */
#include "base.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BASE_POLL_MS 5 // polls every 5ms
#define REPORT_BATCH 128 // limit of up to 128 reports 


// represents base station reply, sent using non blocking MPI 
typedef struct PendingReply {
    BaseReply payload;            // must remain valid until MPI_Isend completes 
    MPI_Request request;    
    struct PendingReply *next; // points to next reply to manage multiple replies concurrenty 
} PendingReply;

typedef struct {
    const Config *cfg;
    MPI_Comm world;
    int base_rank, n_nodes;
    MpiTypes types;
    FILE *log, *timing;
    double started_at;

    /* All timestamps are taken locally at the base (wsn_time, CLOCK_MONOTONIC). */
    double *last_report_at;       /* -1 means this rank has never reported */
    double *last_observed_at;     /* latest availability snapshot seen in a report */
    int *last_observed_avail;

    // stores event data from simulation
    PendingReply *pending_replies;
    long reports_received;
    long replies_sent;
    long terminations_sent;
    long nbr_msgs_reported_sent;
    long nbr_msgs_reported_recv;
} Base;


/**
 * Function that is used to check if an event happened recently (e.g. if a neighbouring node
 * of a requesting node also sent a request recently)
 */
static int is_recent(double timestamp, double now, double period)
{
    return timestamp >= 0.0 && now - timestamp <= period;
}

/* Retire completed nonblocking replies and release their send buffers. */
static void progress_replies(Base *b)
{
    PendingReply **link = &b->pending_replies;
    while (*link) {
        PendingReply *p = *link;
        int done = 0;
        MPI_Test(&p->request, &done, MPI_STATUS_IGNORE);
        if (done) {
            *link = p->next;
            free(p);
        } else {
            link = &p->next;
        }
    }
}

/**
 * Function that updates base station's info on a charging node (last observed date and num port available)
 */
static void remember_availability(Base *b, int rank, int avail, double now)
{
    if (rank < 0 || rank >= b->n_nodes) return;
    b->last_observed_at[rank] = now;
    b->last_observed_avail[rank] = avail;
}

/*
 * The task describes a two-hop search: Node 0 checks nodes 2, 4 and 6
 * after finding its immediate neighbours (1 and 3) congested.
 *
 * A missing recent congestion report is treated as an indicator that a
 * node MAY be available.
 */
static int find_candidates(const Base *b, int origin, double now,
                           int ranks[MAX_SUGGEST], int coords[MAX_SUGGEST][2])
{
    // gets grid measurements
    const int rows = b->cfg->m;
    const int cols = b->cfg->n;
    const int row = origin / cols;
    const int col = origin % cols;
    const double T = b->cfg->period_s;
    // neighbour vectors
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    /* If every  existing direct neighbour reported recently, return none. */
    int neighbours = 0, recent_neighbour_reports = 0;
    for (int i = 0; i < 4; i++) {
        int r = row + dr[i], c = col + dc[i];
        // skip if outside of grid (for nodes on edges)
        if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
        neighbours++;
        if (is_recent(b->last_report_at[r * cols + c], now, T))
            recent_neighbour_reports++;
    }
    // if all neighbours have reported recently, then no available nodes
    if (neighbours == 0 || recent_neighbour_reports == neighbours)
        return 0;

    // checks for neighbours of the reporting nodes neighbours 
    int count = 0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int distance = abs(r - row) + abs(c - col);
            if (distance != 2) continue; 
            int rank = r * cols + c;

            /* A recent congestion report means this candidate is unsuitable. */
            if (is_recent(b->last_report_at[rank], now, T)) continue;

            /* Also exclude known recent below-threshold availability. */
            if (is_recent(b->last_observed_at[rank], now, T) &&
                b->last_observed_avail[rank] < b->cfg->threshold)
                continue;
            
            // adds candidates as long as count is below max suggestion
            if (count < MAX_SUGGEST) {
                ranks[count] = rank;
                coords[count][0] = r;
                coords[count][1] = c;
                count++;
            }
        }
    }
    /* All selected candidates are tied at Manhattan distance 2. */
    return count;
}

/**
 * Function that handles the response of base to charging node with the new node the car should go to
 */
static void queue_base_reply(Base *b, int destination, const Report *report,
                             double now)
{
    // allocated reply space, if not enough then abort
    PendingReply *p = calloc(1, sizeof *p);
    if (!p) {
        fprintf(stderr, "base: out of memory for reply\n");
        MPI_Abort(b->world, 1);
    }

    // gets incident sequence number for id, and report send time so node can calculate RTT
    p->payload.seq = report->seq;
    p->payload.report_sent_at = report->sent_at;

    // gets the nodes available 
    p->payload.n_available = find_candidates(b, destination, now,
                                              p->payload.ranks, p->payload.coords);
    p->payload.sent_at = wsn_time();
    
    // non blocking send
    MPI_Isend(&p->payload, 1, b->types.base_reply, destination,
              TAG_BASE_REPLY, b->world, &p->request);

    // add to pending replies
    p->next = b->pending_replies;
    b->pending_replies = p;
    b->replies_sent++;

    // reports results
    fprintf(b->log, "  recommendation for node %d, report #%d: ",
            destination, report->seq);
    if (p->payload.n_available == 0) fprintf(b->log, "NO nearby nodes available");
    for (int i = 0; i < p->payload.n_available; i++) {
        fprintf(b->log, "%snode %d (%d,%d)", i ? ", " : "",
                p->payload.ranks[i], p->payload.coords[i][0],
                p->payload.coords[i][1]);
    }
    fputc('\n', b->log);
}

/**
 * Function to process congestion report 
 */
static void handle_report(Base *b, int source)
{

    Report report;

    // blocking receive, but called after Iprobe, so only called when a report arrives
    MPI_Recv(&report, 1, b->types.report, source, TAG_REPORT,
             b->world, MPI_STATUS_IGNORE);

    
    // update base stations's data on the reporting node's dats
    const double now = wsn_time();
    const double elapsed = now - b->started_at;

    b->reports_received++;
    b->last_report_at[source] = now;
    remember_availability(b, source, report.avail, now);

    /* Learn about other nodes through the reporter's neighbour snapshots. */
    for (int i = 0; i < MAX_NBRS; i++) {
        if (report.nbr_replied[i] && report.nbr_rank[i] >= 0)
            remember_availability(b, report.nbr_rank[i], report.nbr_avail[i], now);
    }
    b->nbr_msgs_reported_sent += report.inc_msgs_sent;
    b->nbr_msgs_reported_recv += report.inc_msgs_recv;


    // format and log report
    char timestamp[40];
    format_timestamp(report.ts, timestamp, sizeof timestamp);
    fprintf(b->log,
            "\n[t=%.3f] ALERT #%ld: source=%d, seq=%d, coord=(%d,%d), "
            "discovered=%s, free=%d/%d, threshold=%d\n",
            elapsed, b->reports_received, source, report.seq,
            report.coord[0], report.coord[1], timestamp,
            report.avail, report.k, report.threshold);
    fprintf(b->log, "  ports: ");
    for (int i = 0; i < report.k && i < MAX_PORTS; i++)
        fprintf(b->log, "%s%d", i ? "," : "", report.ports[i]);
    fprintf(b->log,
            "\n  neighbour exchange: msgs sent=%d recv=%d, bytes sent=%d recv=%d, "
            "exchange_time=%.6f s\n",
            report.inc_msgs_sent, report.inc_msgs_recv,
            report.inc_bytes_sent, report.inc_bytes_recv, report.exchange_time);
    fprintf(b->log,
            "  cumulative neighbour traffic: msgs sent=%d recv=%d, bytes sent=%d recv=%d\n",
            report.tot_msgs_sent, report.tot_msgs_recv,
            report.tot_bytes_sent, report.tot_bytes_recv);

    for (int i = 0; i < MAX_NBRS; i++) {
        if (report.nbr_rank[i] < 0) continue;
        fprintf(b->log,
                "  %-6s neighbour: node %d (%d,%d), replied=%d, free=%d, RTT=%.6f s\n",
                NBR_NAME[i], report.nbr_rank[i],
                report.nbr_coord[i][0], report.nbr_coord[i][1],
                report.nbr_replied[i], report.nbr_avail[i], report.nbr_rtt[i]);
    }

    /* one-way delay is exact when node and base share a host; across hosts use the RTT the node logs */
    fprintf(b->timing, "REPORT,%d,%d,%d,%zu,%.9f,-1,%.6f\n",
            b->base_rank, source, report.seq, sizeof report,
            now - report.sent_at, elapsed);

    if (report.rank != source)
        fprintf(b->log, "  WARNING: report.rank=%d but MPI source=%d\n",
                report.rank, source);
    queue_base_reply(b, source, &report, now);
    fflush(b->log);
    fflush(b->timing);
}

/* Take a bounded batch so the duration timer is checked regularly. */
static int receive_reports(Base *b)
{
    int received = 0;
    for (int i = 0; i < REPORT_BATCH; i++) {
        int has_report = 0;
        MPI_Status status;
        // probes for reports
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_REPORT, b->world,
                   &has_report, &status);
        if (!has_report) break;
        // if report received, validate and handle
        if (status.MPI_SOURCE >= 0 && status.MPI_SOURCE < b->n_nodes) {
            handle_report(b, status.MPI_SOURCE);
            received++;
        }
    }
    return received;
}

int base_main(const Config *cfg, MPI_Comm world, int n_nodes)
{
    // intialise
    Base b;
    memset(&b, 0, sizeof b);

    // stores configuration, communicator, num nodes, starting time and base ranl
    b.cfg = cfg;
    b.world = world;
    b.n_nodes = n_nodes;
    b.started_at = wsn_time();
    MPI_Comm_rank(world, &b.base_rank);

    // allocate memory to report-history arrays, abort if not enough
    b.last_report_at = malloc((size_t)n_nodes * sizeof(double));
    b.last_observed_at = malloc((size_t)n_nodes * sizeof(double));
    b.last_observed_avail = malloc((size_t)n_nodes * sizeof(int));
    if (!b.last_report_at || !b.last_observed_at || !b.last_observed_avail) {
        fprintf(stderr, "base: out of memory for report history\n");
        MPI_Abort(world, 1);
    }

    //intialise report-history entries for each node
    for (int i = 0; i < n_nodes; i++) {
        b.last_report_at[i] = -1.0;
        b.last_observed_at[i] = -1.0;
        b.last_observed_avail[i] = -1;
    }
    mpi_types_create(&b.types);

    // opens logging files and writes report
    char path[512];
    snprintf(path, sizeof path, "%s/base.log", cfg->log_dir);
    b.log = fopen(path, "w");
    snprintf(path, sizeof path, "%s/base_timing.csv", cfg->log_dir);
    b.timing = fopen(path, "w");
    if (!b.log || !b.timing) {
        fprintf(stderr, "base: cannot open files in %s\n", cfg->log_dir);
        MPI_Abort(world, 1);
    }
    fprintf(b.log, "# EV charging base station (rank %d)\n", b.base_rank);
    config_dump(cfg, b.log);
    fprintf(b.log, "# Two-hop recommendation; absence of a report is not proof of free ports.\n");
    fprintf(b.timing, "msg_type,my_rank,peer,seq,bytes,one_way_delay_s,rtt_s,t_rel_s\n");

    printf("[base %d] listening for reports from %d nodes for %.2f s (T=%.2f s)\n",
           b.base_rank, n_nodes, cfg->duration_s, cfg->period_s);
    fflush(stdout);
    
    // main sumulation loop, runs until specified duration
    while (wsn_time() - b.started_at < cfg->duration_s) {
        int received = receive_reports(&b);
        progress_replies(&b);
        if (!received) sleep_ms(BASE_POLL_MS);
    }

    /* Send shutdown to EVERY EV node; keep receiving any in-flight reports. */
    double termination_sent_at = wsn_time();
    MPI_Request *terminations = malloc((size_t)n_nodes * sizeof *terminations);
    if (!terminations) {
        fprintf(stderr, "base: out of memory for termination requests\n");
        MPI_Abort(world, 1);
    }
    // send termination messages
    for (int i = 0; i < n_nodes; i++) {
        MPI_Isend(&termination_sent_at, 1, MPI_DOUBLE, i,
                  TAG_TERMINATE, world, &terminations[i]);
        b.terminations_sent++;
    }

    // runs while each node receives termination and terminates. while waiting, base station
    // handles reportd and processes outstanding replies
    int remaining = n_nodes;
    while (remaining > 0) {
        receive_reports(&b);
        progress_replies(&b);
        for (int i = 0; i < n_nodes; i++) {
            if (terminations[i] == MPI_REQUEST_NULL) continue;
            int done = 0;
            MPI_Test(&terminations[i], &done, MPI_STATUS_IGNORE);
            if (done) remaining--;
        }
        if (remaining > 0) sleep_ms(BASE_POLL_MS);
    }
    free(terminations);

    /* Progress any buffered reply sends before releasing their buffers. */
    while (b.pending_replies) {
        progress_replies(&b);
        if (b.pending_replies) sleep_ms(BASE_POLL_MS);
    }

    fprintf(b.log, "\n# ---- summary ----\n");
    fprintf(b.log, "# simulation time                 : %.3f s\n", wsn_time() - b.started_at);
    fprintf(b.log, "# alerts / reports received       : %ld\n", b.reports_received);
    fprintf(b.log, "# base replies sent               : %ld\n", b.replies_sent);
    fprintf(b.log, "# termination messages sent       : %ld\n", b.terminations_sent);
    fprintf(b.log, "# base messages/events total      : %ld\n",
            b.reports_received + b.replies_sent + b.terminations_sent);
    fprintf(b.log, "# incident neighbour msgs sent    : %ld\n", b.nbr_msgs_reported_sent);
    fprintf(b.log, "# incident neighbour msgs received: %ld\n", b.nbr_msgs_reported_recv);
    fprintf(b.log, "# shutdown                        : graceful\n");

    printf("[base %d] finished: %ld alerts, %ld replies, %ld terminations\n",
           b.base_rank, b.reports_received, b.replies_sent, b.terminations_sent);
    fflush(stdout);
    fclose(b.log);
    fclose(b.timing);
    mpi_types_free(&b.types);
    free(b.last_report_at);
    free(b.last_observed_at);
    free(b.last_observed_avail);
    return 0;
}
