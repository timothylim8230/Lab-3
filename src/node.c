/*
 * node.c - EV charging node (Task 1)
 * FIT3143 Lab 3
 *
 * One MPI process per node. Inside the process an OpenMP team of k+1 threads:
 *   thread 0      : communication thread, the ONLY thread that calls MPI
 *                   (MPI_THREAD_FUNNELED). Runs the cycle loop, neighbour
 *                   exchange, base station reporting and shutdown.
 *   threads 1..k  : one charging port each. Periodically flips the port's
 *                   in-use/free status in the shared array port_status[].
 *
 * All MPI traffic in the comm thread is non-blocking or probe-then-receive, so
 * a node is never stuck in MPI_Recv: neighbour requests and the termination
 * message are serviced while waiting for anything else.
 */
#define _POSIX_C_SOURCE 200809L
#include "node.h"

#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPLY_POOL 64         /* outstanding NbrReply sends kept alive until complete */
#define IDLE_SLEEP_MS 5       /* comm thread sleep between service passes */
#define WAIT_SLEEP_MS 1       /* sleep while waiting for neighbour / base replies */
#define PORT_SLEEP_MS 20      /* port thread sleep slice (so it notices stop_flag quickly) */
#define NBR_TIMEOUT_MULT 2.0  /* neighbour reply timeout = mult * cycle_s */
#define BASE_TIMEOUT_MULT 5.0 /* base reply timeout = mult * cycle_s */

typedef struct
{
    int ts[6];
    int avail;
} CycleRecord;

typedef struct
{
    /* identity / topology */
    int rank, cart_rank, coord[2];
    int nbr[MAX_NBRS]; /* cart ranks, MPI_PROC_NULL if absent */
    int nbr_coord[MAX_NBRS][2];
    int n_nbrs;
    MPI_Comm cart, world;
    int base_rank;
    const Config *cfg;
    MpiTypes types;

    /* shared with the port threads */
    int port_status[MAX_PORTS]; /* 1 = in use, 0 = free */
    volatile int stop_flag;

    /* history ring buffer (Table 1 style) */
    CycleRecord history[HISTORY_LEN];
    int hist_len, hist_head;

    /* counters for neighbour traffic (cumulative) */
    int tot_msgs_sent, tot_msgs_recv, tot_bytes_sent, tot_bytes_recv;
    int seq; /* incident counter */
    int n_requests_served;
    int n_redirects, n_reports, n_base_replies, n_nbr_timeouts;

    /* outstanding NbrReply sends */
    struct
    {
        int active;
        MPI_Request req;
        NbrReply buf;
    } pool[REPLY_POOL];

    /* outstanding base station reply */
    int base_pending;
    MPI_Request base_req;
    BaseReply base_buf;
    double base_sent_at;
    int base_seq;

    /* logs */
    FILE *log, *timing;
    double t_start;
} Node;

/* ------------------------------------------------------------------------ */
/* logging helpers                                                          */
/* ------------------------------------------------------------------------ */

static void logf_ts(Node *nd, const char *fmt, ...)
{
    int ts[6];
    char tbuf[32];
    va_list ap;
    wall_timestamp(ts);
    format_timestamp(ts, tbuf, sizeof tbuf);
    fprintf(nd->log, "[%s | t=%8.3f] ", tbuf, MPI_Wtime() - nd->t_start);
    va_start(ap, fmt);
    vfprintf(nd->log, fmt, ap);
    va_end(ap);
    fputc('\n', nd->log);
}

/* one row per received message for the Task 3 analysis */
static void timing_row(Node *nd, const char *msg_type, int peer, int seq, size_t bytes,
                       double one_way, double rtt)
{
    fprintf(nd->timing, "%s,%d,%d,%d,%zu,%.9f,%.9f,%.6f\n",
            msg_type, nd->rank, peer, seq, bytes, one_way, rtt, MPI_Wtime() - nd->t_start);
}

/* ------------------------------------------------------------------------ */
/* shared array access                                                      */
/* ------------------------------------------------------------------------ */

/* Comm thread: consistent-per-element snapshot of the shared array. */
static int snapshot_ports(Node *nd, int out[MAX_PORTS])
{
    int avail = 0;
    for (int p = 0; p < nd->cfg->k; p++)
    {
        int v;
#pragma omp atomic read
        v = nd->port_status[p];
        out[p] = v;
        if (!v)
            avail++;
    }
    return avail;
}

static void history_push(Node *nd, const int ts[6], int avail)
{
    CycleRecord *r = &nd->history[nd->hist_head];
    memcpy(r->ts, ts, sizeof r->ts);
    r->avail = avail;
    nd->hist_head = (nd->hist_head + 1) % HISTORY_LEN;
    if (nd->hist_len < HISTORY_LEN)
        nd->hist_len++;
}

/* ------------------------------------------------------------------------ */
/* port thread (threads 1..k)                                               */
/* ------------------------------------------------------------------------ */

static void port_thread(Node *nd, int port)
{
    unsigned s = nd->cfg->seed * 7919u + (unsigned)nd->rank * 131u + (unsigned)port * 17u + 1u;
    const double p = nd->cfg->congestion_prob;
    const double cycle = nd->cfg->cycle_s;

    /* initial status */
    int status = ((double)rand_r(&s) / (double)RAND_MAX) < p;
#pragma omp atomic write
    nd->port_status[port] = status;

    /* random phase so ports do not all flip at the same instant (EVs arrive independently) */
    double next = omp_get_wtime() + cycle * ((double)rand_r(&s) / (double)RAND_MAX);

    while (!nd->stop_flag)
    {
        if (omp_get_wtime() < next)
        {
            sleep_ms(PORT_SLEEP_MS);
            continue;
        }
        next += cycle;
        status = ((double)rand_r(&s) / (double)RAND_MAX) < p; /* EV arrives (1) or departs (0) */
#pragma omp atomic write
        nd->port_status[port] = status;
    }
}

/* ------------------------------------------------------------------------ */
/* comm thread helpers                                                      */
/* ------------------------------------------------------------------------ */

static int pool_slot(Node *nd)
{
    for (;;)
    {
        for (int i = 0; i < REPLY_POOL; i++)
        {
            if (!nd->pool[i].active)
                return i;
            int done;
            MPI_Test(&nd->pool[i].req, &done, MPI_STATUS_IGNORE);
            if (done)
            {
                nd->pool[i].active = 0;
                return i;
            }
        }
        sleep_ms(WAIT_SLEEP_MS); /* all 64 slots busy: extremely unlikely, wait a little */
    }
}

static void pool_drain(Node *nd)
{
    for (int i = 0; i < REPLY_POOL; i++)
    {
        if (nd->pool[i].active)
        {
            MPI_Wait(&nd->pool[i].req, MPI_STATUS_IGNORE);
            nd->pool[i].active = 0;
        }
    }
}

/* Answer one neighbour request that is already known to be waiting. */
static void serve_nbr_request(Node *nd, const MPI_Status *st)
{
    NbrRequest req;
    MPI_Recv(&req, 1, nd->types.nbr_req, st->MPI_SOURCE, TAG_NBR_REQ, nd->cart, MPI_STATUS_IGNORE);
    double now = MPI_Wtime();
    nd->tot_msgs_recv++;
    nd->tot_bytes_recv += (int)sizeof(NbrRequest);
    nd->n_requests_served++;
    timing_row(nd, "NBR_REQ", req.src_rank, req.seq, sizeof(NbrRequest), now - req.sent_at, -1.0);

    int slot = pool_slot(nd);
    NbrReply *rep = &nd->pool[slot].buf;
    memset(rep, 0, sizeof *rep);
    rep->src_rank = nd->cart_rank;
    rep->seq = req.seq;
    rep->k = nd->cfg->k;
    rep->avail = snapshot_ports(nd, rep->ports);
    rep->req_sent_at = req.sent_at;
    rep->sent_at = MPI_Wtime();
    MPI_Isend(rep, 1, nd->types.nbr_reply, st->MPI_SOURCE, TAG_NBR_REPLY, nd->cart, &nd->pool[slot].req);
    nd->pool[slot].active = 1;
    nd->tot_msgs_sent++;
    nd->tot_bytes_sent += (int)sizeof(NbrReply);

    logf_ts(nd, "served availability request #%d from neighbour node %d: replied avail=%d/%d",
            req.seq, req.src_rank, rep->avail, rep->k);
}

static void handle_base_reply(Node *nd)
{
    double now = MPI_Wtime();
    BaseReply *r = &nd->base_buf;
    nd->base_pending = 0;
    nd->n_base_replies++;
    timing_row(nd, "BASE_REPLY", nd->base_rank, r->seq, sizeof(BaseReply), now - r->sent_at, now - r->report_sent_at);

    if (r->n_available == 0)
    {
        logf_ts(nd, "base station reply for report #%d: NO nodes available nearby (rtt=%.6f s). Forwarding to EVs: please wait.",
                r->seq, now - r->report_sent_at);
        printf("[node %d (%d,%d)] base station: no nearby nodes available for report #%d\n",
               nd->rank, nd->coord[0], nd->coord[1], r->seq);
    }
    else
    {
        char list[256] = "";
        size_t used = 0;
        for (int i = 0; i < r->n_available && i < MAX_SUGGEST; i++)
            used += (size_t)snprintf(list + used, sizeof list - used, "%s%d(%d,%d)", i ? ", " : "",
                                     r->ranks[i], r->coords[i][0], r->coords[i][1]);
        logf_ts(nd, "base station reply for report #%d: %d nearest available node(s): %s (rtt=%.6f s). Forwarding to EVs: redirect to %s.",
                r->seq, r->n_available, list, now - r->report_sent_at, list);
        printf("[node %d (%d,%d)] base station suggests nearest available nodes for report #%d: %s\n",
               nd->rank, nd->coord[0], nd->coord[1], r->seq, list);
    }
    fflush(stdout);
}

/*
 * Service pass: run frequently from everywhere the comm thread waits.
 *   1. termination from the base station
 *   2. availability requests from neighbours (answered immediately => no deadlock
 *      when two neighbours are congested at the same time)
 *   3. outstanding base station reply
 * Returns 1 once the termination message has been received.
 */
static int service(Node *nd)
{
    int flag;
    MPI_Status st;

    MPI_Iprobe(nd->base_rank, TAG_TERMINATE, nd->world, &flag, &st);
    if (flag)
    {
        double sent_at, now;
        MPI_Recv(&sent_at, 1, MPI_DOUBLE, nd->base_rank, TAG_TERMINATE, nd->world, MPI_STATUS_IGNORE);
        now = MPI_Wtime();
        timing_row(nd, "TERMINATE", nd->base_rank, 0, sizeof(double), now - sent_at, -1.0);
        logf_ts(nd, "termination message received from base station (delay %.6f s); starting graceful shutdown", now - sent_at);
        nd->stop_flag = 1;
#pragma omp flush
        return 1;
    }

    for (;;)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_NBR_REQ, nd->cart, &flag, &st);
        if (!flag)
            break;
        serve_nbr_request(nd, &st);
    }

    if (nd->base_pending)
    {
        MPI_Test(&nd->base_req, &flag, MPI_STATUS_IGNORE);
        if (flag)
        {
            if (nd->base_buf.seq == nd->base_seq)
            {
                handle_base_reply(nd);
            }
            else
            {
                /* stale reply for an earlier timed-out report: discard and keep waiting */
                logf_ts(nd, "discarded stale base station reply for report #%d", nd->base_buf.seq);
                MPI_Irecv(&nd->base_buf, 1, nd->types.base_reply, nd->base_rank, TAG_BASE_REPLY, nd->world, &nd->base_req);
            }
        }
        else if (MPI_Wtime() - nd->base_sent_at > BASE_TIMEOUT_MULT * nd->cfg->cycle_s)
        {
            logf_ts(nd, "WARNING: no base station reply for report #%d after %.1f s; giving up on it",
                    nd->base_seq, BASE_TIMEOUT_MULT * nd->cfg->cycle_s);
            MPI_Cancel(&nd->base_req);
            MPI_Wait(&nd->base_req, MPI_STATUS_IGNORE);
            nd->base_pending = 0;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* congestion handling                                                      */
/* ------------------------------------------------------------------------ */

/*
 * Availability dropped below the threshold:
 *   1. ask every existing neighbour for its availability (non-blocking)
 *   2. wait for the replies while still servicing other traffic
 *   3. if a neighbour has vacancies -> redirect incoming EVs there
 *      else                         -> report the congestion incident to the base station
 * Returns 1 if termination arrived while waiting.
 */
static int handle_congestion(Node *nd, int avail, const int ports[MAX_PORTS], const int ts[6])
{
    const Config *cfg = nd->cfg;
    const int seq = ++nd->seq;
    char tbuf[32];
    format_timestamp(ts, tbuf, sizeof tbuf);

    logf_ts(nd, "CONGESTION detected (incident #%d): availability %d/%d < threshold %d -> requesting data from %d neighbour(s)",
            seq, avail, cfg->k, cfg->threshold, nd->n_nbrs);

    NbrRequest req = {.src_rank = nd->cart_rank, .seq = seq, .sent_at = 0.0};
    NbrReply rep[MAX_NBRS];
    MPI_Request sreq[MAX_NBRS], rreq[MAX_NBRS];
    int pending[MAX_NBRS] = {0}, got[MAX_NBRS] = {0};
    double rtt[MAX_NBRS] = {-1, -1, -1, -1};
    int remaining = 0, terminated = 0;
    int inc_sent = 0, inc_recv = 0, inc_bytes_sent = 0, inc_bytes_recv = 0;

    /* 1. send requests + post receives for the replies */
    double t_req = MPI_Wtime();
    req.sent_at = t_req;
    for (int i = 0; i < MAX_NBRS; i++)
    {
        if (nd->nbr[i] == MPI_PROC_NULL)
            continue;
        MPI_Irecv(&rep[i], 1, nd->types.nbr_reply, nd->nbr[i], TAG_NBR_REPLY, nd->cart, &rreq[i]);
        MPI_Isend(&req, 1, nd->types.nbr_req, nd->nbr[i], TAG_NBR_REQ, nd->cart, &sreq[i]);
        pending[i] = 1;
        remaining++;
        inc_sent++;
        inc_bytes_sent += (int)sizeof(NbrRequest);
    }

    /* 2. wait for replies, servicing other traffic meanwhile */
    const double timeout = NBR_TIMEOUT_MULT * cfg->cycle_s;
    while (remaining > 0 && MPI_Wtime() - t_req < timeout)
    {
        for (int i = 0; i < MAX_NBRS; i++)
        {
            if (!pending[i])
                continue;
            int done;
            MPI_Test(&rreq[i], &done, MPI_STATUS_IGNORE);
            if (!done)
                continue;
            if (rep[i].seq != seq)
            {
                /* late reply to an earlier timed-out request: discard, re-post */
                logf_ts(nd, "discarded stale reply (incident #%d) from neighbour %d", rep[i].seq, rep[i].src_rank);
                MPI_Irecv(&rep[i], 1, nd->types.nbr_reply, nd->nbr[i], TAG_NBR_REPLY, nd->cart, &rreq[i]);
                continue;
            }
            double now = MPI_Wtime();
            pending[i] = 0;
            got[i] = 1;
            remaining--;
            rtt[i] = now - rep[i].req_sent_at;
            inc_recv++;
            inc_bytes_recv += (int)sizeof(NbrReply);
            timing_row(nd, "NBR_REPLY", rep[i].src_rank, seq, sizeof(NbrReply), now - rep[i].sent_at, rtt[i]);
            logf_ts(nd, "  reply from %s neighbour node %d (%d,%d): avail=%d/%d  rtt=%.6f s",
                    NBR_NAME[i], nd->nbr[i], nd->nbr_coord[i][0], nd->nbr_coord[i][1], rep[i].avail, rep[i].k, rtt[i]);
        }
        if (remaining > 0)
        {
            if (service(nd))
            {
                terminated = 1;
                break;
            }
            sleep_ms(WAIT_SLEEP_MS);
        }
    }
    double exchange_time = MPI_Wtime() - t_req;

    /* cancel receives that never completed (neighbour slow or already terminated) */
    for (int i = 0; i < MAX_NBRS; i++)
    {
        if (!pending[i])
            continue;
        MPI_Cancel(&rreq[i]);
        MPI_Wait(&rreq[i], MPI_STATUS_IGNORE);
        nd->n_nbr_timeouts++;
        if (!terminated)
            logf_ts(nd, "  WARNING: no reply from %s neighbour node %d within %.1f s", NBR_NAME[i], nd->nbr[i], timeout);
    }
    for (int i = 0; i < MAX_NBRS; i++)
        if (nd->nbr[i] != MPI_PROC_NULL)
            MPI_Wait(&sreq[i], MPI_STATUS_IGNORE);

    nd->tot_msgs_sent += inc_sent;
    nd->tot_msgs_recv += inc_recv;
    nd->tot_bytes_sent += inc_bytes_sent;
    nd->tot_bytes_recv += inc_bytes_recv;

    if (terminated)
        return 1;

    /* 3. decide: redirect to a neighbour with vacancies, or report to the base station */
    int best = -1;
    for (int i = 0; i < MAX_NBRS; i++)
        if (got[i] && rep[i].avail >= cfg->threshold && (best < 0 || rep[i].avail > rep[best].avail))
            best = i;

    if (best >= 0)
    {
        nd->n_redirects++;
        logf_ts(nd, "incident #%d resolved locally: redirecting incoming EVs to %s neighbour node %d (%d,%d) with %d free port(s)",
                seq, NBR_NAME[best], nd->nbr[best], nd->nbr_coord[best][0], nd->nbr_coord[best][1], rep[best].avail);
        printf("[node %d (%d,%d)] congested (%d/%d free) -> redirecting EVs to neighbour %d (%d free)\n",
               nd->rank, nd->coord[0], nd->coord[1], avail, cfg->k, nd->nbr[best], rep[best].avail);
        fflush(stdout);
        return 0;
    }

    /* all neighbours congested (or silent): report to the base station */
    if (nd->base_pending)
    {
        /* previous report still unanswered; drop it so the new one can be tracked */
        logf_ts(nd, "previous report #%d still unanswered; cancelling its receive", nd->base_seq);
        MPI_Cancel(&nd->base_req);
        MPI_Wait(&nd->base_req, MPI_STATUS_IGNORE);
        nd->base_pending = 0;
    }

    Report rp;
    memset(&rp, 0, sizeof rp);
    rp.seq = seq;
    rp.rank = nd->rank;
    rp.cart_rank = nd->cart_rank;
    rp.coord[0] = nd->coord[0];
    rp.coord[1] = nd->coord[1];
    memcpy(rp.ts, ts, sizeof rp.ts);
    rp.k = cfg->k;
    rp.avail = avail;
    rp.threshold = cfg->threshold;
    memcpy(rp.ports, ports, sizeof(int) * (size_t)cfg->k);
    rp.n_nbrs = nd->n_nbrs;
    for (int i = 0; i < MAX_NBRS; i++)
    {
        rp.nbr_rank[i] = (nd->nbr[i] == MPI_PROC_NULL) ? -1 : nd->nbr[i];
        rp.nbr_coord[i][0] = nd->nbr_coord[i][0];
        rp.nbr_coord[i][1] = nd->nbr_coord[i][1];
        rp.nbr_replied[i] = got[i];
        rp.nbr_avail[i] = got[i] ? rep[i].avail : -1;
        rp.nbr_rtt[i] = rtt[i];
    }
    rp.inc_msgs_sent = inc_sent;
    rp.inc_msgs_recv = inc_recv;
    rp.inc_bytes_sent = inc_bytes_sent;
    rp.inc_bytes_recv = inc_bytes_recv;
    rp.tot_msgs_sent = nd->tot_msgs_sent;
    rp.tot_msgs_recv = nd->tot_msgs_recv;
    rp.tot_bytes_sent = nd->tot_bytes_sent;
    rp.tot_bytes_recv = nd->tot_bytes_recv;
    rp.exchange_time = exchange_time;

    /* post the receive first so the reply can never be missed, then send */
    MPI_Irecv(&nd->base_buf, 1, nd->types.base_reply, nd->base_rank, TAG_BASE_REPLY, nd->world, &nd->base_req);
    nd->base_pending = 1;
    nd->base_seq = seq;
    rp.sent_at = MPI_Wtime();
    nd->base_sent_at = rp.sent_at;
    MPI_Request srep;
    MPI_Isend(&rp, 1, nd->types.report, nd->base_rank, TAG_REPORT, nd->world, &srep);
    MPI_Wait(&srep, MPI_STATUS_IGNORE); /* small message: completes as soon as it is buffered */
    nd->n_reports++;

    logf_ts(nd, "incident #%d: all neighbours congested -> REPORT sent to base station "
                "(discovered %s, avail=%d/%d, nbr msgs sent=%d recv=%d, bytes sent=%d recv=%d, exchange %.6f s, report %zu B)",
            seq, tbuf, avail, cfg->k, inc_sent, inc_recv, inc_bytes_sent, inc_bytes_recv, exchange_time, sizeof(Report));
    printf("[node %d (%d,%d)] congested (%d/%d free), all neighbours congested -> report #%d sent to base station\n",
           nd->rank, nd->coord[0], nd->coord[1], avail, cfg->k, seq);
    fflush(stdout);
    return 0;
}

/* ------------------------------------------------------------------------ */
/* comm thread main loop (thread 0)                                         */
/* ------------------------------------------------------------------------ */

static void comm_thread(Node *nd)
{
    const Config *cfg = nd->cfg;
    int ports[MAX_PORTS], ts[6];
    char tbuf[32];
    int terminated = 0;

    /* first cycle record immediately so the log is never empty */
    double next_cycle = MPI_Wtime();

    while (!terminated)
    {
        if (service(nd))
        {
            terminated = 1;
            break;
        }

        double now = MPI_Wtime();
        if (now < next_cycle)
        {
            sleep_ms(IDLE_SLEEP_MS);
            continue;
        }
        next_cycle += cfg->cycle_s;

        int avail = snapshot_ports(nd, ports);
        wall_timestamp(ts);
        history_push(nd, ts, avail);
        format_timestamp(ts, tbuf, sizeof tbuf);

        /* Table 1 style row */
        fprintf(nd->log, "%04d %02d %02d %02d %02d %02d  avail=%2d/%d  ports=[",
                ts[0], ts[1], ts[2], ts[3], ts[4], ts[5], avail, cfg->k);
        for (int p = 0; p < cfg->k; p++)
            fprintf(nd->log, "%s%d", p ? " " : "", ports[p]);
        fprintf(nd->log, "]\n");

        if (avail < cfg->threshold)
        {
            if (handle_congestion(nd, avail, ports, ts))
                terminated = 1;
        }
        fflush(nd->log);
    }

    /* ---- graceful shutdown of the comm side ---- */
    nd->stop_flag = 1;
#pragma omp flush
    if (nd->base_pending)
    {
        MPI_Cancel(&nd->base_req);
        MPI_Wait(&nd->base_req, MPI_STATUS_IGNORE);
        nd->base_pending = 0;
    }
    pool_drain(nd);
}

/* ------------------------------------------------------------------------ */
/* entry point                                                              */
/* ------------------------------------------------------------------------ */

int node_main(const Config *cfg, MPI_Comm node_comm, MPI_Comm world, int base_rank)
{
    Node *nd = calloc(1, sizeof *nd);
    if (!nd)
    {
        fprintf(stderr, "node: out of memory\n");
        return 1;
    }
    nd->cfg = cfg;
    nd->world = world;
    nd->base_rank = base_rank;
    nd->t_start = MPI_Wtime();
    MPI_Comm_rank(world, &nd->rank);

    /* ---- MPI virtual topology: m x n Cartesian grid, non-periodic, no reorder ---- */
    int dims[2] = {cfg->m, cfg->n};
    int periods[2] = {0, 0};
    MPI_Cart_create(node_comm, 2, dims, periods, /*reorder=*/0, &nd->cart);
    MPI_Comm_rank(nd->cart, &nd->cart_rank);
    MPI_Cart_coords(nd->cart, nd->cart_rank, 2, nd->coord);
    MPI_Cart_shift(nd->cart, 0, 1, &nd->nbr[NBR_TOP], &nd->nbr[NBR_BOTTOM]); /* along rows    */
    MPI_Cart_shift(nd->cart, 1, 1, &nd->nbr[NBR_LEFT], &nd->nbr[NBR_RIGHT]); /* along columns */
    for (int i = 0; i < MAX_NBRS; i++)
    {
        if (nd->nbr[i] == MPI_PROC_NULL)
        {
            nd->nbr_coord[i][0] = nd->nbr_coord[i][1] = -1;
            continue;
        }
        MPI_Cart_coords(nd->cart, nd->nbr[i], 2, nd->nbr_coord[i]);
        nd->n_nbrs++;
    }
    mpi_types_create(&nd->types);

    /* ---- logs ---- */
    char path[512];
    snprintf(path, sizeof path, "%s/node_%d.log", cfg->log_dir, nd->rank);
    nd->log = fopen(path, "w");
    snprintf(path, sizeof path, "%s/node_%d_timing.csv", cfg->log_dir, nd->rank);
    nd->timing = fopen(path, "w");
    if (!nd->log || !nd->timing)
    {
        fprintf(stderr, "node %d: cannot open log files in %s\n", nd->rank, cfg->log_dir);
        if (nd->log)
            fclose(nd->log);
        if (nd->timing)
            fclose(nd->timing);
        mpi_types_free(&nd->types);
        MPI_Comm_free(&nd->cart);
        free(nd);
        return 1;
    }
    fprintf(nd->log, "# EV charging node %d\n", nd->rank);
    config_dump(cfg, nd->log);
    fprintf(nd->log, "# cart rank %d, coord (%d,%d), neighbours: top=%d bottom=%d left=%d right=%d\n",
            nd->cart_rank, nd->coord[0], nd->coord[1],
            nd->nbr[NBR_TOP], nd->nbr[NBR_BOTTOM], nd->nbr[NBR_LEFT], nd->nbr[NBR_RIGHT]);
    fprintf(nd->log, "# Year Mon Day Hr Min Sec  availability  port status (1=in use, 0=free)\n");
    fprintf(nd->timing, "msg_type,my_rank,peer,seq,bytes,one_way_delay_s,rtt_s,t_rel_s\n");

    printf("Global rank: %d. Cart rank: %d. Coord: (%d, %d). Left: %d. Right: %d. Top: %d. Bottom: %d\n",
           nd->rank, nd->cart_rank, nd->coord[0], nd->coord[1],
           nd->nbr[NBR_LEFT], nd->nbr[NBR_RIGHT], nd->nbr[NBR_TOP], nd->nbr[NBR_BOTTOM]);
    fflush(stdout);

    /* ---- OpenMP team: thread 0 = comm, threads 1..k = charging ports ---- */
    omp_set_dynamic(0);
#pragma omp parallel num_threads(cfg->k + 1) default(none) shared(nd, cfg)
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        if (tid == 0)
        {
            if (nth < cfg->k + 1)
                fprintf(nd->log, "# WARNING: only %d OpenMP threads available, ports are shared between threads\n", nth);
            comm_thread(nd); /* only this thread touches MPI */
        }
        else
        {
            /* normally one port per thread; if fewer threads were granted, share */
            for (int p = tid - 1; p < cfg->k; p += nth - 1)
                port_thread(nd, p);
        }
    }

    /* ---- summary + cleanup ---- */
    fprintf(nd->log, "# ---- summary ----\n");
    fprintf(nd->log, "# cycles recorded          : %d\n", nd->hist_len);
    fprintf(nd->log, "# congestion incidents     : %d\n", nd->seq);
    fprintf(nd->log, "#   resolved by redirect   : %d\n", nd->n_redirects);
    fprintf(nd->log, "#   reported to base       : %d (replies received: %d)\n", nd->n_reports, nd->n_base_replies);
    fprintf(nd->log, "# neighbour requests served: %d\n", nd->n_requests_served);
    fprintf(nd->log, "# neighbour reply timeouts : %d\n", nd->n_nbr_timeouts);
    fprintf(nd->log, "# neighbour msgs sent/recv : %d / %d  (bytes %d / %d)\n",
            nd->tot_msgs_sent, nd->tot_msgs_recv, nd->tot_bytes_sent, nd->tot_bytes_recv);
    fprintf(nd->log, "# run time                 : %.3f s\n", MPI_Wtime() - nd->t_start);
    fprintf(nd->log, "# shutdown                 : graceful (termination message from base station)\n");
    fclose(nd->log);
    fclose(nd->timing);

    printf("[node %d (%d,%d)] shut down gracefully: %d incidents, %d redirects, %d reports\n",
           nd->rank, nd->coord[0], nd->coord[1], nd->seq, nd->n_redirects, nd->n_reports);
    fflush(stdout);

    mpi_types_free(&nd->types);
    MPI_Comm_free(&nd->cart);
    free(nd);
    return 0;
}
