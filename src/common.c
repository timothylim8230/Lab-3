/*
 * common.c - MPI derived datatypes and small helpers
 * FIT3143 Lab 3
 */
#define _POSIX_C_SOURCE 200809L
/**
 * Timothy Lim  33111472 tlim0034@student.monash.edu
 * Scott Nguyen 33879095 sngu0065@student.monash.edu
 */
#include "common.h"

#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

const char *const NBR_NAME[MAX_NBRS] = {"top", "bottom", "left", "right"};

/*
 * Every message struct is laid out as "ints first, then doubles", so a single
 * two-block MPI_Type_create_struct describes it. The type is resized to the
 * C struct size so arrays of the struct would also work.
 */
static MPI_Datatype make_int_double_type(size_t first_double_off, size_t struct_size)
{
    int blocklens[2];
    MPI_Aint displs[2];
    MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
    MPI_Datatype tmp, out;

    blocklens[0] = (int)(first_double_off / sizeof(int));
    blocklens[1] = (int)((struct_size - first_double_off) / sizeof(double));
    displs[0] = 0;
    displs[1] = (MPI_Aint)first_double_off;

    MPI_Type_create_struct(2, blocklens, displs, types, &tmp);
    MPI_Type_create_resized(tmp, 0, (MPI_Aint)struct_size, &out);
    MPI_Type_commit(&out);
    MPI_Type_free(&tmp);
    return out;
}

void mpi_types_create(MpiTypes *t)
{
    t->nbr_req = make_int_double_type(offsetof(NbrRequest, sent_at), sizeof(NbrRequest));
    t->nbr_reply = make_int_double_type(offsetof(NbrReply, req_sent_at), sizeof(NbrReply));
    t->report = make_int_double_type(offsetof(Report, nbr_rtt), sizeof(Report));
    t->base_reply = make_int_double_type(offsetof(BaseReply, report_sent_at), sizeof(BaseReply));
}

void mpi_types_free(MpiTypes *t)
{
    MPI_Type_free(&t->nbr_req);
    MPI_Type_free(&t->nbr_reply);
    MPI_Type_free(&t->report);
    MPI_Type_free(&t->base_reply);
}

double wsn_time(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

void wall_timestamp(int out[6])
{
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    out[0] = tm.tm_year + 1900;
    out[1] = tm.tm_mon + 1;
    out[2] = tm.tm_mday;
    out[3] = tm.tm_hour;
    out[4] = tm.tm_min;
    out[5] = tm.tm_sec;
}

void format_timestamp(const int ts[6], char *buf, size_t n)
{
    snprintf(buf, n, "%04d-%02d-%02d %02d:%02d:%02d", ts[0], ts[1], ts[2], ts[3], ts[4], ts[5]);
}

void sleep_ms(double ms)
{
    struct timespec req;
    if (ms <= 0)
        return;
    req.tv_sec = (time_t)(ms / 1000.0);
    req.tv_nsec = (long)((ms - (double)req.tv_sec * 1000.0) * 1e6);
    nanosleep(&req, NULL);
}

int ensure_dir(const char *path)
{
    if (mkdir(path, 0775) == 0)
        return 0;
    return (errno == EEXIST) ? 0 : -1;
}

void config_dump(const Config *cfg, FILE *f)
{
    fprintf(f, "# grid            : %d x %d (%d nodes + 1 base station)\n", cfg->m, cfg->n, cfg->m * cfg->n);
    fprintf(f, "# ports per node  : %d\n", cfg->k);
    fprintf(f, "# threshold       : %d free ports (congested if availability < threshold)\n", cfg->threshold);
    fprintf(f, "# update cycle    : %.2f s\n", cfg->cycle_s);
    fprintf(f, "# base period T   : %.2f s\n", cfg->period_s);
    fprintf(f, "# duration        : %.2f s\n", cfg->duration_s);
    fprintf(f, "# seed            : %u\n", cfg->seed);
    fprintf(f, "# congestion prob : %.2f\n", cfg->congestion_prob);
    fprintf(f, "# comm version    : %d\n", COMM_VERSION);
    fprintf(f, "# message sizes   : NbrRequest=%zu B, NbrReply=%zu B, Report=%zu B, BaseReply=%zu B, Terminate=%zu B\n",
            sizeof(NbrRequest), sizeof(NbrReply), sizeof(Report), sizeof(BaseReply), sizeof(double));
}
