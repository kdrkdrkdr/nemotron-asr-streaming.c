/*
 * nemotron_asr_kernels.c - compute kernels, dispatch, and threading.
 * Persistent worker pool (pthread / Win32) with output-row parallelism, the
 * typed-weight linear/matvec/argmax dispatchers (F32 and packed Q8P int8 with
 * dynamic activation quantization), fused QKV / prompt / LSTM-gate paths, and
 * scalar layer-norm/softmax/activation helpers. Architecture-specific hot
 * kernels are selected in nemotron_asr_kernels_impl.h.
 */
#include "nemotron_asr.h"
#include "nemotron_asr_kernels_impl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#include <sys/time.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif
#endif

/*
 * Thread / sync abstraction: pthread on POSIX, Win32 SRWLOCK + condition
 * variables on Windows. SRWLOCK/CONDITION_VARIABLE both support static
 * initializers, so the global pool struct keeps the same shape on both.
 */
#ifdef _WIN32
typedef HANDLE nemo_thread_t;
typedef SRWLOCK nemo_mutex_t;
typedef CONDITION_VARIABLE nemo_cond_t;
#define NEMO_MUTEX_INIT SRWLOCK_INIT
#define NEMO_COND_INIT CONDITION_VARIABLE_INIT
#define NEMO_MUTEX_LOCK(m) AcquireSRWLockExclusive(m)
#define NEMO_MUTEX_UNLOCK(m) ReleaseSRWLockExclusive(m)
#define NEMO_COND_WAIT(c, m) SleepConditionVariableSRW((c), (m), INFINITE, 0)
#define NEMO_COND_SIGNAL(c) WakeConditionVariable(c)
#define NEMO_COND_BROADCAST(c) WakeAllConditionVariable(c)
#define NEMO_THREAD_RET unsigned __stdcall
#define NEMO_THREAD_RETURN_VAL 0u
#else
typedef pthread_t nemo_thread_t;
typedef pthread_mutex_t nemo_mutex_t;
typedef pthread_cond_t nemo_cond_t;
#define NEMO_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER
#define NEMO_COND_INIT PTHREAD_COND_INITIALIZER
#define NEMO_MUTEX_LOCK(m) pthread_mutex_lock(m)
#define NEMO_MUTEX_UNLOCK(m) pthread_mutex_unlock(m)
#define NEMO_COND_WAIT(c, m) pthread_cond_wait((c), (m))
#define NEMO_COND_SIGNAL(c) pthread_cond_signal(c)
#define NEMO_COND_BROADCAST(c) pthread_cond_broadcast(c)
#define NEMO_THREAD_RET void *
#define NEMO_THREAD_RETURN_VAL NULL
#endif

#define NEMO_MAX_THREADS 16
#define NEMO_PARALLEL_WORK_MIN 262144
#define NEMO_Q8_STACK_MAX 4096
#define NEMO_Q8_PREQ_STACK_MAX 65536
#define NEMO_Q8_SCALE_STACK_MAX 512

typedef void (*nemo_parallel_fn_t)(int tid, int n_threads, void *arg);

static struct {
    nemo_thread_t threads[NEMO_MAX_THREADS - 1];
    int tids[NEMO_MAX_THREADS - 1];
    int n_threads;
    int shutdown;

    nemo_parallel_fn_t fn;
    void *arg;
    int generation;

    nemo_mutex_t mutex;
    nemo_cond_t cond_work;
    nemo_cond_t cond_done;
    int n_done;
} g_tp = {
    .n_threads = 1,
    .shutdown = 0,
    .generation = 0,
    .mutex = NEMO_MUTEX_INIT,
    .cond_work = NEMO_COND_INIT,
    .cond_done = NEMO_COND_INIT,
};

static NEMO_THREAD_RET nemo_worker_loop(void *arg) {
    int tid = *(int *)arg;
    int my_gen = 0;

    for (;;) {
        NEMO_MUTEX_LOCK(&g_tp.mutex);
        while (g_tp.generation == my_gen && !g_tp.shutdown) {
            NEMO_COND_WAIT(&g_tp.cond_work, &g_tp.mutex);
        }
        if (g_tp.shutdown) {
            NEMO_MUTEX_UNLOCK(&g_tp.mutex);
            return NEMO_THREAD_RETURN_VAL;
        }
        my_gen = g_tp.generation;
        nemo_parallel_fn_t fn = g_tp.fn;
        void *a = g_tp.arg;
        int nt = g_tp.n_threads;
        NEMO_MUTEX_UNLOCK(&g_tp.mutex);

        fn(tid, nt, a);

        NEMO_MUTEX_LOCK(&g_tp.mutex);
        if (++g_tp.n_done >= g_tp.n_threads - 1) {
            NEMO_COND_SIGNAL(&g_tp.cond_done);
        }
        NEMO_MUTEX_UNLOCK(&g_tp.mutex);
    }
}

static int nemo_thread_create(nemo_thread_t *t, int *tid_arg) {
#ifdef _WIN32
    uintptr_t h = _beginthreadex(NULL, 0, nemo_worker_loop, tid_arg, 0, NULL);
    if (h == 0) return -1;
    *t = (HANDLE)h;
    return 0;
#else
    return pthread_create(t, NULL, nemo_worker_loop, tid_arg);
#endif
}

static void nemo_thread_join(nemo_thread_t t) {
#ifdef _WIN32
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
#else
    pthread_join(t, NULL);
#endif
}

static void nemo_parallel_for(nemo_parallel_fn_t fn, void *arg) {
    if (g_tp.n_threads <= 1) {
        fn(0, 1, arg);
        return;
    }

    NEMO_MUTEX_LOCK(&g_tp.mutex);
    g_tp.fn = fn;
    g_tp.arg = arg;
    g_tp.n_done = 0;
    g_tp.generation++;
    NEMO_COND_BROADCAST(&g_tp.cond_work);
    NEMO_MUTEX_UNLOCK(&g_tp.mutex);

    fn(0, g_tp.n_threads, arg);

    NEMO_MUTEX_LOCK(&g_tp.mutex);
    while (g_tp.n_done < g_tp.n_threads - 1) {
        NEMO_COND_WAIT(&g_tp.cond_done, &g_tp.mutex);
    }
    NEMO_MUTEX_UNLOCK(&g_tp.mutex);
}

void nemo_set_threads(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads > NEMO_MAX_THREADS) n_threads = NEMO_MAX_THREADS;
    if (n_threads == g_tp.n_threads) return;

    if (g_tp.n_threads > 1) {
        NEMO_MUTEX_LOCK(&g_tp.mutex);
        g_tp.shutdown = 1;
        NEMO_COND_BROADCAST(&g_tp.cond_work);
        NEMO_MUTEX_UNLOCK(&g_tp.mutex);
        for (int i = 0; i < g_tp.n_threads - 1; i++) {
            nemo_thread_join(g_tp.threads[i]);
        }
        NEMO_MUTEX_LOCK(&g_tp.mutex);
        g_tp.shutdown = 0;
        g_tp.generation = 0;
        g_tp.fn = NULL;
        g_tp.arg = NULL;
        g_tp.n_done = 0;
        NEMO_MUTEX_UNLOCK(&g_tp.mutex);
    }

    g_tp.n_threads = n_threads;
    if (n_threads <= 1) return;

    for (int i = 0; i < n_threads - 1; i++) {
        g_tp.tids[i] = i + 1;
        if (nemo_thread_create(&g_tp.threads[i], &g_tp.tids[i]) != 0) {
            g_tp.n_threads = i + 1;
            return;
        }
    }
}

static int nemo_weight_q8_stride(const nemo_weight_t *w, int fallback) {
    return (w && w->q8_stride > 0) ? (int)w->q8_stride : fallback;
}

static float nemo_quantize_q8_symmetric_padded(const float *x, int8_t *x_q8,
                                               int n, int stride) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(x[i]);
        if (v > max_abs) max_abs = v;
    }
    if (max_abs <= 0.0f) {
        memset(x_q8, 0, (size_t)stride);
        return 0.0f;
    }
    float scale = max_abs / 127.0f;
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        float qf = x[i] * inv;
        int q = (int)(qf >= 0.0f ? qf + 0.5f : qf - 0.5f);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        x_q8[i] = (int8_t)q;
    }
    if (stride > n) memset(x_q8 + n, 0, (size_t)(stride - n));
    return scale;
}

static int8_t *nemo_q8_tmp_alloc(int n, int8_t *stack_buf) {
    if (n <= NEMO_Q8_STACK_MAX) return stack_buf;
    return (int8_t *)malloc((size_t)n);
}

static void nemo_q8_tmp_free(int8_t *ptr, int8_t *stack_buf) {
    if (ptr != stack_buf) free(ptr);
}

static float *nemo_f32_tmp_alloc(int n, float *stack_buf, int stack_max) {
    if (n <= stack_max) return stack_buf;
    return (float *)malloc((size_t)n * sizeof(float));
}

static void nemo_f32_tmp_free(float *ptr, float *stack_buf) {
    if (ptr != stack_buf) free(ptr);
}

typedef struct {
    const float *x;
    int8_t *x_q8;
    float *x_scales;
    int rows;
    int in_dim;
    int q8_stride;
} nemo_q8_prequant_task_t;

static void nemo_q8_prequant_worker(int tid, int n_threads, void *arg) {
    nemo_q8_prequant_task_t *t = (nemo_q8_prequant_task_t *)arg;
    int start = (t->rows * tid) / n_threads;
    int end = (t->rows * (tid + 1)) / n_threads;
    for (int r = start; r < end; r++) {
        t->x_scales[r] = nemo_quantize_q8_symmetric_padded(t->x + (size_t)r * t->in_dim,
                                                           t->x_q8 + (size_t)r * t->q8_stride,
                                                           t->in_dim, t->q8_stride);
    }
}

static void nemo_q8_prequant_rows(const float *x, int8_t *x_q8, float *x_scales,
                                  int rows, int in_dim, int q8_stride) {
    nemo_q8_prequant_task_t task = {x, x_q8, x_scales, rows, in_dim, q8_stride};
    long long work = (long long)rows * (long long)in_dim;
    if (g_tp.n_threads > 1 && rows > 1 && work >= 8192) {
        nemo_parallel_for(nemo_q8_prequant_worker, &task);
    } else {
        nemo_q8_prequant_worker(0, 1, &task);
    }
}

/*
 * Q8P packed-weight addressing. int8 weights are stored in tiles of 4 output
 * rows with layout [row_tile][col_block][4 rows][16 cols], where `stride` is
 * the input dimension padded up to a multiple of 16. For output row `row` and
 * input column block starting at `k` (a multiple of 16), this returns the
 * 16-byte int8 group covering columns [k, k+16): tile = row>>2 picks the 4-row
 * tile, lane = row&3 picks the row within it, and (k>>4)*64 steps over column
 * blocks (4 rows x 16 cols = 64 bytes each). The converter writes this exact
 * layout (tools/convert_nemo.py: tensor_q8p_bytes); the generic/NEON/AVX
 * backends reimplement the same addressing inline.
 */
static const int8_t *nemo_q8p_row_block(const int8_t *w, int row, int stride, int k) {
    int tile = row >> 2;
    int lane = row & 3;
    return w + (size_t)tile * 4u * (size_t)stride +
           (size_t)(k >> 4) * 64u + (size_t)lane * 16u;
}

static int32_t nemo_dot_i8_i8_packed_scalar(const int8_t *x_q8, const int8_t *w,
                                            int row, int stride) {
    int32_t sum = 0;
    for (int k = 0; k < stride; k += 16) {
        const int8_t *wr = nemo_q8p_row_block(w, row, stride, k);
        for (int j = 0; j < 16; j++) sum += (int32_t)x_q8[k + j] * (int32_t)wr[j];
    }
    return sum;
}

static int8_t nemo_q8p_value(const int8_t *w, int row, int stride, int col) {
    return nemo_q8p_row_block(w, row, stride, col & ~15)[col & 15];
}

static float nemo_q8_dot_quantized_row(const int8_t *x_q8, float x_scale,
                                       const nemo_weight_t *w, int row, int stride) {
    int q8_stride = nemo_weight_q8_stride(w, stride);
    int32_t acc = nemo_dot_i8_i8_packed_scalar(x_q8, w->q8, row, q8_stride);
    return (float)acc * x_scale * w->q8_scales[row];
}

static float nemo_weight_value_row(const nemo_weight_t *w, int row, int stride, int col) {
    int8_t qv = nemo_q8p_value(w->q8, row, nemo_weight_q8_stride(w, stride), col);
    return (float)qv * w->q8_scales[row];
}

/* Payload shared by the Q8P linear workers. */
typedef struct {
    float *y;
    const float *x;
    const nemo_weight_t *w;
    const float *b;
    int rows;
    int in_dim;
    int out_dim;
} nemo_linear_weight_task_t;

typedef struct {
    float *y;
    const int8_t *x_q8;
    const float *x_scales;
    const nemo_weight_t *w;
    const float *b;
    int rows;
    int in_dim;
    int q8_stride;
    int out_dim;
} nemo_q8_linear_preq_task_t;

static void nemo_q8_linear_worker(int tid, int n_threads, void *arg) {
    nemo_linear_weight_task_t *t = (nemo_linear_weight_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    if (start >= end) return;
    int q8_stride = nemo_weight_q8_stride(t->w, t->in_dim);
    int8_t stack_q8[NEMO_Q8_STACK_MAX];
    int8_t *x_q8 = nemo_q8_tmp_alloc(q8_stride, stack_q8);
    if (!x_q8) return;
    for (int r = 0; r < t->rows; r++) {
        const float *xr = t->x + (size_t)r * t->in_dim;
        float x_scale = nemo_quantize_q8_symmetric_padded(xr, x_q8, t->in_dim, q8_stride);
        nemo_q8p_matvec_fused_impl(t->y + (size_t)r * t->out_dim + start,
                                   x_q8, x_scale, t->w->q8, t->w->q8_scales,
                                   t->b ? t->b + start : NULL,
                                   q8_stride, start, end - start);
    }
    nemo_q8_tmp_free(x_q8, stack_q8);
}

static void nemo_q8_linear_preq_worker(int tid, int n_threads, void *arg) {
    nemo_q8_linear_preq_task_t *t = (nemo_q8_linear_preq_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    if (start >= end) return;
    for (int r = 0; r < t->rows; r++) {
        const int8_t *x_q8 = t->x_q8 + (size_t)r * t->q8_stride;
        float x_scale = t->x_scales[r];
        nemo_q8p_matvec_fused_impl(t->y + (size_t)r * t->out_dim + start,
                                   x_q8, x_scale, t->w->q8, t->w->q8_scales,
                                   t->b ? t->b + start : NULL,
                                   t->q8_stride, start, end - start);
    }
}

/* Payload shared by the Q8P fused-QKV workers. */
typedef struct {
    float *y0;
    float *y1;
    float *y2;
    const float *x;
    const nemo_weight_t *w0;
    const nemo_weight_t *w1;
    const nemo_weight_t *w2;
    int rows;
    int in_dim;
    int out_dim;
} nemo_linear3_weight_task_t;

typedef struct {
    float *y0;
    float *y1;
    float *y2;
    const int8_t *x_q8;
    const float *x_scales;
    const nemo_weight_t *w0;
    const nemo_weight_t *w1;
    const nemo_weight_t *w2;
    int rows;
    int in_dim;
    int q8_stride;
    int q8_stride0;
    int q8_stride1;
    int q8_stride2;
    int out_dim;
} nemo_q8_linear3_preq_task_t;

static void nemo_q8_linear3_worker(int tid, int n_threads, void *arg) {
    nemo_linear3_weight_task_t *t = (nemo_linear3_weight_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    if (start >= end) return;
    int q8_stride0 = nemo_weight_q8_stride(t->w0, t->in_dim);
    int q8_stride1 = nemo_weight_q8_stride(t->w1, t->in_dim);
    int q8_stride2 = nemo_weight_q8_stride(t->w2, t->in_dim);
    int q8_stride = q8_stride0;
    if (q8_stride1 > q8_stride) q8_stride = q8_stride1;
    if (q8_stride2 > q8_stride) q8_stride = q8_stride2;
    int8_t stack_q8[NEMO_Q8_STACK_MAX];
    int8_t *x_q8 = nemo_q8_tmp_alloc(q8_stride, stack_q8);
    if (!x_q8) return;
    for (int r = 0; r < t->rows; r++) {
        const float *xr = t->x + (size_t)r * t->in_dim;
        float x_scale = nemo_quantize_q8_symmetric_padded(xr, x_q8, t->in_dim, q8_stride);
        nemo_q8p_matvec_fused_impl(t->y0 + (size_t)r * t->out_dim + start,
                                   x_q8, x_scale, t->w0->q8, t->w0->q8_scales,
                                   NULL, q8_stride0, start, end - start);
        nemo_q8p_matvec_fused_impl(t->y1 + (size_t)r * t->out_dim + start,
                                   x_q8, x_scale, t->w1->q8, t->w1->q8_scales,
                                   NULL, q8_stride1, start, end - start);
        nemo_q8p_matvec_fused_impl(t->y2 + (size_t)r * t->out_dim + start,
                                   x_q8, x_scale, t->w2->q8, t->w2->q8_scales,
                                   NULL, q8_stride2, start, end - start);
    }
    nemo_q8_tmp_free(x_q8, stack_q8);
}

static void nemo_q8_linear3_preq_worker(int tid, int n_threads, void *arg) {
    nemo_q8_linear3_preq_task_t *t = (nemo_q8_linear3_preq_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    if (start >= end) return;
    for (int r = 0; r < t->rows; r++) {
        const int8_t *x_q8 = t->x_q8 + (size_t)r * t->q8_stride;
        float x_scale = t->x_scales[r];
        nemo_q8p_matvec_fused_impl(t->y0 + (size_t)r * t->out_dim + start,
                                   x_q8, x_scale, t->w0->q8, t->w0->q8_scales,
                                   NULL, t->q8_stride0, start, end - start);
        nemo_q8p_matvec_fused_impl(t->y1 + (size_t)r * t->out_dim + start,
                                   x_q8, x_scale, t->w1->q8, t->w1->q8_scales,
                                   NULL, t->q8_stride1, start, end - start);
        nemo_q8p_matvec_fused_impl(t->y2 + (size_t)r * t->out_dim + start,
                                   x_q8, x_scale, t->w2->q8, t->w2->q8_scales,
                                   NULL, t->q8_stride2, start, end - start);
    }
}

typedef struct {
    const float *x;
    const nemo_weight_t *w;
    const float *b;
    const int8_t *x_q8;
    float x_scale;
    int in_dim;
    int q8_stride;
    int out_dim;
    int best[NEMO_MAX_THREADS];
    float best_val[NEMO_MAX_THREADS];
} nemo_argmax_weight_task_t;

static void nemo_argmax_weight_worker(int tid, int n_threads, void *arg) {
    nemo_argmax_weight_task_t *t = (nemo_argmax_weight_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    int8_t stack_q8[NEMO_Q8_STACK_MAX];
    int q8_stride = nemo_weight_q8_stride(t->w, t->in_dim);
    const int8_t *x_q8 = t->x_q8;
    float x_scale = t->x_scale;
    int8_t *tmp_q8 = NULL;
    if (!x_q8) {
        tmp_q8 = nemo_q8_tmp_alloc(q8_stride, stack_q8);
        if (tmp_q8) {
            x_scale = nemo_quantize_q8_symmetric_padded(t->x, tmp_q8, t->in_dim, q8_stride);
            x_q8 = tmp_q8;
        }
    }
    if (!x_q8) {
        t->best[tid] = start;
        t->best_val[tid] = -3.4028234663852886e38f;
        return;
    }
    t->best[tid] = nemo_argmax_q8p_range_impl(x_q8, x_scale,
                                              t->w->q8, t->w->q8_scales, t->b,
                                              q8_stride, start, end,
                                              &t->best_val[tid]);
    if (tmp_q8) nemo_q8_tmp_free(tmp_q8, stack_q8);
}

typedef struct {
    float *y;
    const float *x;
    const nemo_weight_t *w;
    const float *b;
    int rows;
    int in_dim;
    int prompt_dim;
    int prompt_id;
    int out_dim;
} nemo_prompt_q8_task_t;

typedef struct {
    float *y;
    const float *x;
    const float *h;
    const nemo_weight_t *w_ih;
    const nemo_weight_t *w_hh;
    const float *b_ih;
    const float *b_hh;
    int dim;
    int out_dim;
} nemo_lstm_gates_q8_task_t;

static void nemo_prompt_q8_worker(int tid, int n_threads, void *arg) {
    nemo_prompt_q8_task_t *t = (nemo_prompt_q8_task_t *)arg;
    int stride = t->in_dim + t->prompt_dim;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    if (start >= end) return;
    int q8_stride = nemo_weight_q8_stride(t->w, stride);
    int8_t stack_q8[NEMO_Q8_STACK_MAX];
    int8_t *x_q8 = nemo_q8_tmp_alloc(q8_stride, stack_q8);
    if (!x_q8) return;
    for (int r = 0; r < t->rows; r++) {
        const float *xr = t->x + (size_t)r * t->in_dim;
        float x_scale = nemo_quantize_q8_symmetric_padded(xr, x_q8, t->in_dim, q8_stride);
        for (int o = start; o < end; o++) {
            float v = (t->b ? t->b[o] : 0.0f) +
                      nemo_weight_value_row(t->w, o, stride, t->in_dim + t->prompt_id);
            v += nemo_q8_dot_quantized_row(x_q8, x_scale, t->w, o, stride);
            t->y[(size_t)r * t->out_dim + o] = v > 0.0f ? v : 0.0f;
        }
    }
    nemo_q8_tmp_free(x_q8, stack_q8);
}

static void nemo_lstm_gates_q8_worker(int tid, int n_threads, void *arg) {
    nemo_lstm_gates_q8_task_t *t = (nemo_lstm_gates_q8_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    if (start >= end) return;
    int x_q8_stride = nemo_weight_q8_stride(t->w_ih, t->dim);
    int h_q8_stride = nemo_weight_q8_stride(t->w_hh, t->dim);
    int8_t stack_x_q8[NEMO_Q8_STACK_MAX];
    int8_t stack_h_q8[NEMO_Q8_STACK_MAX];
    int8_t *x_q8 = nemo_q8_tmp_alloc(x_q8_stride, stack_x_q8);
    int8_t *h_q8 = nemo_q8_tmp_alloc(h_q8_stride, stack_h_q8);
    if (!x_q8 || !h_q8) {
        if (x_q8) nemo_q8_tmp_free(x_q8, stack_x_q8);
        if (h_q8) nemo_q8_tmp_free(h_q8, stack_h_q8);
        return;
    }
    float x_scale = nemo_quantize_q8_symmetric_padded(t->x, x_q8, t->dim, x_q8_stride);
    float h_scale = nemo_quantize_q8_symmetric_padded(t->h, h_q8, t->dim, h_q8_stride);
    for (int o = start; o < end; o++) {
        float v = 0.0f;
        if (t->b_ih) v += t->b_ih[o];
        if (t->b_hh) v += t->b_hh[o];
        v += nemo_q8_dot_quantized_row(x_q8, x_scale, t->w_ih, o, t->dim);
        v += nemo_q8_dot_quantized_row(h_q8, h_scale, t->w_hh, o, t->dim);
        t->y[o] = v;
    }
    nemo_q8_tmp_free(x_q8, stack_x_q8);
    nemo_q8_tmp_free(h_q8, stack_h_q8);
}

float *nemo_alloc(size_t count, size_t elem) {
    if (count == 0 || elem == 0 || count > (SIZE_MAX / elem)) return NULL;
    float *p = (float *)calloc(count, elem);
    if (!p) {
        fprintf(stderr, "nemotron: allocation failed (%zu x %zu)\n", count, elem);
    }
    return p;
}

double nemo_time_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, ctr;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&ctr);
    return (double)ctr.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
#endif
}

void nemo_linear_weight(float *y, const float *x, const nemo_weight_t *w, const float *b,
                        int rows, int in_dim, int out_dim) {
    long long work = (long long)rows * (long long)in_dim * (long long)out_dim;
    nemo_linear_weight_task_t task = {y, x, w, b, rows, in_dim, out_dim};
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        if (rows == 1) {
            int q8_stride = nemo_weight_q8_stride(w, in_dim);
            int8_t stack_q8[NEMO_Q8_PREQ_STACK_MAX];
            float stack_scales[NEMO_Q8_SCALE_STACK_MAX];
            int8_t *x_q8 = q8_stride <= NEMO_Q8_PREQ_STACK_MAX ? stack_q8 : (int8_t *)malloc((size_t)q8_stride);
            float *x_scales = nemo_f32_tmp_alloc(1, stack_scales, NEMO_Q8_SCALE_STACK_MAX);
            if (x_q8 && x_scales) {
                nemo_q8_prequant_rows(x, x_q8, x_scales, rows, in_dim, q8_stride);
                nemo_q8_linear_preq_task_t preq = {y, x_q8, x_scales, w, b,
                                                   rows, in_dim, q8_stride, out_dim};
                nemo_parallel_for(nemo_q8_linear_preq_worker, &preq);
                if (x_q8 != stack_q8) free(x_q8);
                nemo_f32_tmp_free(x_scales, stack_scales);
                return;
            }
            if (x_q8 && x_q8 != stack_q8) free(x_q8);
            if (x_scales) nemo_f32_tmp_free(x_scales, stack_scales);
        }
        nemo_parallel_for(nemo_q8_linear_worker, &task);
        return;
    }
    nemo_q8_linear_worker(0, 1, &task);
}

void nemo_linear_nobias_weight(float *y, const float *x, const nemo_weight_t *w,
                               int rows, int in_dim, int out_dim) {
    nemo_linear_weight(y, x, w, NULL, rows, in_dim, out_dim);
}

void nemo_linear3_nobias_weight(float *y0, float *y1, float *y2, const float *x,
                                const nemo_weight_t *w0, const nemo_weight_t *w1,
                                const nemo_weight_t *w2,
                                int rows, int in_dim, int out_dim) {
    long long work = (long long)rows * (long long)in_dim * (long long)out_dim * 3LL;
    nemo_linear3_weight_task_t task = {y0, y1, y2, x, w0, w1, w2, rows, in_dim, out_dim};
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        if (rows == 1) {
            int q8_stride0 = nemo_weight_q8_stride(w0, in_dim);
            int q8_stride1 = nemo_weight_q8_stride(w1, in_dim);
            int q8_stride2 = nemo_weight_q8_stride(w2, in_dim);
            int q8_stride = q8_stride0;
            if (q8_stride1 > q8_stride) q8_stride = q8_stride1;
            if (q8_stride2 > q8_stride) q8_stride = q8_stride2;
            int8_t stack_q8[NEMO_Q8_PREQ_STACK_MAX];
            float stack_scales[NEMO_Q8_SCALE_STACK_MAX];
            int8_t *x_q8 = q8_stride <= NEMO_Q8_PREQ_STACK_MAX ? stack_q8 : (int8_t *)malloc((size_t)q8_stride);
            float *x_scales = nemo_f32_tmp_alloc(1, stack_scales, NEMO_Q8_SCALE_STACK_MAX);
            if (x_q8 && x_scales) {
                nemo_q8_prequant_rows(x, x_q8, x_scales, rows, in_dim, q8_stride);
                nemo_q8_linear3_preq_task_t preq = {y0, y1, y2, x_q8, x_scales,
                                                    w0, w1, w2, rows, in_dim,
                                                    q8_stride, q8_stride0,
                                                    q8_stride1, q8_stride2, out_dim};
                nemo_parallel_for(nemo_q8_linear3_preq_worker, &preq);
                if (x_q8 != stack_q8) free(x_q8);
                nemo_f32_tmp_free(x_scales, stack_scales);
                return;
            }
            if (x_q8 && x_q8 != stack_q8) free(x_q8);
            if (x_scales) nemo_f32_tmp_free(x_scales, stack_scales);
        }
        nemo_parallel_for(nemo_q8_linear3_worker, &task);
        return;
    }
    nemo_q8_linear3_worker(0, 1, &task);
}

void nemo_prompt_linear_relu_weight(float *y, const float *x, const nemo_weight_t *w,
                                    const float *b, int rows, int in_dim,
                                    int prompt_dim, int prompt_id, int out_dim) {
    if (prompt_id < 0 || prompt_id >= prompt_dim) prompt_id = prompt_dim - 1;
    long long work = (long long)rows * (long long)in_dim * (long long)out_dim;
    nemo_prompt_q8_task_t task = {y, x, w, b, rows, in_dim, prompt_dim, prompt_id, out_dim};
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        nemo_parallel_for(nemo_prompt_q8_worker, &task);
        return;
    }
    nemo_prompt_q8_worker(0, 1, &task);
}

void nemo_lstm_gates_weight(float *y, const float *x, const float *h,
                            const nemo_weight_t *w_ih, const nemo_weight_t *w_hh,
                            const float *b_ih, const float *b_hh,
                            int dim, int out_dim) {
    long long work = (long long)dim * (long long)out_dim * 2LL;
    nemo_lstm_gates_q8_task_t task = {y, x, h, w_ih, w_hh, b_ih, b_hh, dim, out_dim};
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        nemo_parallel_for(nemo_lstm_gates_q8_worker, &task);
        return;
    }
    nemo_lstm_gates_q8_worker(0, 1, &task);
}

void nemo_preconv_emit_f32(float *out, const float *history, const float *w, const float *b,
                           int out_start, int out_t, int total_t,
                           int c_in, int c_out, int f_in, int f_out,
                           int k, int stride, int left, int groups) {
    nemo_preconv_emit_f32_impl(out, history, w, b, out_start, out_t, total_t,
                               c_in, c_out, f_in, f_out, k, stride, left, groups);
}

float nemo_dot_f32(const float *a, const float *b, int n) {
    return nemo_dot_f32_impl(a, b, n);
}

float nemo_attention_score_f32(const float *q, const float *bias_u, const float *k,
                               const float *bias_v, const float *p, int n) {
    return nemo_attention_score_f32_impl(q, bias_u, k, bias_v, p, n);
}

int nemo_argmax_matvec_weight(const float *x, const nemo_weight_t *w, const float *b,
                              int in_dim, int out_dim, float *best_val_out) {
    long long work = (long long)in_dim * (long long)out_dim;
    int parallel = g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN;
    nemo_argmax_weight_task_t task;
    memset(&task, 0, sizeof(task));
    task.x = x;
    task.w = w;
    task.b = b;
    task.in_dim = in_dim;
    task.out_dim = out_dim;
    if (parallel) {
        int8_t stack_q8[NEMO_Q8_STACK_MAX];
        int q8_stride = nemo_weight_q8_stride(w, in_dim);
        int8_t *x_q8 = nemo_q8_tmp_alloc(q8_stride, stack_q8);
        if (x_q8) {
            task.x_q8 = x_q8;
            task.x_scale = nemo_quantize_q8_symmetric_padded(x, x_q8, in_dim, q8_stride);
            task.q8_stride = q8_stride;
        }
        nemo_parallel_for(nemo_argmax_weight_worker, &task);
        if (x_q8) nemo_q8_tmp_free(x_q8, stack_q8);
    } else {
        nemo_argmax_weight_worker(0, 1, &task);
    }
    int n_threads = parallel ? g_tp.n_threads : 1;
    int best = 0;
    float best_val = -3.4028234663852886e38f;
    for (int i = 0; i < n_threads; i++) {
        if (task.best_val[i] > best_val) {
            best_val = task.best_val[i];
            best = task.best[i];
        }
    }
    if (best_val_out) *best_val_out = best_val;
    return best;
}

void nemo_fft512_power_f32(float *power, const float *frame) {
    nemo_fft512_power_f32_impl(power, frame);
}

void nemo_vec_axpy_inplace(float *dst, const float *src, float alpha, int n) {
    nemo_vec_axpy_inplace_impl(dst, src, alpha, n);
}

void nemo_layer_norm(float *y, const float *x, const float *w, const float *b,
                     int rows, int dim, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *yr = y + (size_t)r * dim;
        double mean = 0.0;
        for (int i = 0; i < dim; i++) mean += xr[i];
        mean /= (double)dim;
        double var = 0.0;
        for (int i = 0; i < dim; i++) {
            double d = (double)xr[i] - mean;
            var += d * d;
        }
        float inv = 1.0f / sqrtf((float)(var / (double)dim) + eps);
        for (int i = 0; i < dim; i++) {
            float v = (xr[i] - (float)mean) * inv;
            yr[i] = v * w[i] + b[i];
        }
    }
}

float nemo_sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

void nemo_swish(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] * nemo_sigmoid(x[i]);
}
