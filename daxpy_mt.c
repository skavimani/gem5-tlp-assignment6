/*
 * daxpy_mt.c - Multi-threaded DAXPY kernel (Y = a*X + Y) for gem5 SE mode
 * MSCS-531 Assignment 6: Thread-Level Parallelism with gem5 MinorCPU
 *
 * Usage:   ./daxpy_mt <N elements> <T threads>
 * Example: ./daxpy_mt 65536 4
 *
 * Each thread handles a contiguous chunk of X and Y. The main thread is
 * thread 0 and does its own chunk, so T threads need exactly T CPUs in gem5.
 *
 * Region of interest (ROI): when compiled with -DGEM5, thread 0 resets the
 * gem5 statistics right before the parallel loop and dumps them right after
 * the final barrier (the "annotated portion" of the daxpy kernel). The first
 * block in stats.txt is therefore the parallel DAXPY region only, including
 * the barrier (synchronization) cost.
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef GEM5
#if defined(__x86_64__)
/* gem5 x86 "magic" instruction (0x0F 0x04 <func>), same as libm5's
 * m5_reset_stats / m5_dump_reset_stats, so no libm5 build is needed.
 * func codes from gem5/include/gem5/asm/generic/m5ops.h:
 *   M5OP_RESET_STATS = 0x40, M5OP_DUMP_RESET_STATS = 0x42            */
static inline void m5_reset_stats(unsigned long delay, unsigned long period)
{
    __asm__ volatile(".byte 0x0f, 0x04\n\t.word 0x40"
                     : : "D"(delay), "S"(period) : "rax", "memory");
}
static inline void m5_dump_reset_stats(unsigned long delay,
                                       unsigned long period)
{
    __asm__ volatile(".byte 0x0f, 0x04\n\t.word 0x42"
                     : : "D"(delay), "S"(period) : "rax", "memory");
}
#else
#include <gem5/m5ops.h>   /* other ISAs: link with gem5's libm5.a */
#endif
#endif

static double *X, *Y;
static double alpha = 2.5;
static long N;
static int T;

/* Flag-based spin barrier that needs NO atomic read-modify-write.
 *
 * Why: gem5's MinorCPU (in-order) model does not handle x86 locked
 * instructions (lock xadd / lock cmpxchg) correctly across cores, so
 * pthread_barrier_wait() and an atomic-counter barrier both hung.
 *
 * How: every thread owns one arrival flag (on its own 64-byte cache line
 * to avoid false sharing). Thread 0 waits until all flags reach the
 * current episode number, then publishes a release flag that the other
 * threads spin on. Only ordinary loads and stores are used; release/
 * acquire ordering compiles to plain MOVs on x86 (TSO), and cache
 * coherence makes the flags visible to the other cores.
 * The spinning shows up as cycles/instructions on each simulated core,
 * i.e. the synchronization overhead.                                  */
#define MAX_THREADS 64
#define PAD 8                         /* 8 longs = 64 bytes per flag */
static volatile long arrive[MAX_THREADS * PAD];
static volatile long release_flag[PAD];

static void spin_barrier(int id, long *episode)
{
    long e = ++(*episode);
    if (id == 0) {
        for (int t = 1; t < T; t++)
            while (__atomic_load_n(&arrive[t * PAD], __ATOMIC_ACQUIRE) < e)
                ;   /* spin until thread t arrives */
        __atomic_store_n(&release_flag[0], e, __ATOMIC_RELEASE);
    } else {
        __atomic_store_n(&arrive[id * PAD], e, __ATOMIC_RELEASE);
        while (__atomic_load_n(&release_flag[0], __ATOMIC_ACQUIRE) < e)
            ;   /* spin until thread 0 releases everyone */
    }
}

typedef struct {
    int id;
    long start;
    long end;
} thread_arg_t;

static void *daxpy_worker(void *p)
{
    thread_arg_t *t = (thread_arg_t *)p;
    long episode = 0;

    /* 1) wait until every thread is created and running */
    spin_barrier(t->id, &episode);
#ifdef GEM5
    if (t->id == 0)
        m5_reset_stats(0, 0);          /* ---- ROI begins ---- */
#endif
    /* 2) make sure nobody starts before the stats are reset */
    spin_barrier(t->id, &episode);

    /* ===== DAXPY kernel: this thread's slice ===== */
    for (long i = t->start; i < t->end; i++)
        Y[i] = alpha * X[i] + Y[i];
    /* ============================================ */

    /* 3) wait for the slowest thread (synchronization cost) */
    spin_barrier(t->id, &episode);
#ifdef GEM5
    if (t->id == 0)
        m5_dump_reset_stats(0, 0);     /* ---- ROI ends ---- */
#endif
    return NULL;
}

int main(int argc, char **argv)
{
    N = (argc > 1) ? atol(argv[1]) : 65536;
    T = (argc > 2) ? atoi(argv[2]) : 1;
    if (N <= 0 || T <= 0 || T > MAX_THREADS) {
        fprintf(stderr, "usage: %s <N> <threads>\n", argv[0]);
        return 1;
    }

    X = malloc(N * sizeof(double));
    Y = malloc(N * sizeof(double));
    if (!X || !Y) { fprintf(stderr, "malloc failed\n"); return 1; }

    for (long i = 0; i < N; i++) {     /* deterministic input */
        X[i] = (double)(i % 1000) * 0.5;
        Y[i] = (double)(i % 100);
    }

    pthread_t    tid[T];
    thread_arg_t args[T];

    long chunk = N / T, rem = N % T, pos = 0;
    for (int t = 0; t < T; t++) {      /* static block partitioning */
        long len = chunk + (t < rem ? 1 : 0);
        args[t].id = t;
        args[t].start = pos;
        args[t].end = pos + len;
        pos += len;
    }

    for (int t = 1; t < T; t++)
        pthread_create(&tid[t], NULL, daxpy_worker, &args[t]);
    daxpy_worker(&args[0]);            /* main thread is thread 0 */
    /* No pthread_join(): after the final spin barrier every thread has
     * finished its slice, so Y is complete. Skipping join avoids a futex()
     * wait in gem5; returning from main() ends all threads (exit_group). */

    /* verify against the closed-form result */
    double sum = 0.0;
    long errors = 0;
    for (long i = 0; i < N; i++) {
        double expect = alpha * ((double)(i % 1000) * 0.5) + (double)(i % 100);
        if (Y[i] != expect) errors++;
        sum += Y[i];
    }
    printf("DAXPY done: N=%ld threads=%d checksum=%.1f errors=%ld\n",
           N, T, sum, errors);

    free(X);
    free(Y);
    return errors ? 2 : 0;
}
