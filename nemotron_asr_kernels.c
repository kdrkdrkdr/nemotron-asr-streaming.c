#include "nemotron_asr.h"
#include "nemotron_asr_kernels_impl.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif
#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#define NEMO_MAX_THREADS 16
#define NEMO_PARALLEL_WORK_MIN 262144
#define NEMO_BLAS_ROWS_MIN 16
#define NEMO_Q8_STACK_MAX 4096

typedef void (*nemo_parallel_fn_t)(int tid, int n_threads, void *arg);

static struct {
    pthread_t threads[NEMO_MAX_THREADS - 1];
    int tids[NEMO_MAX_THREADS - 1];
    int n_threads;
    int shutdown;

    nemo_parallel_fn_t fn;
    void *arg;
    int generation;

    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
    int n_done;
} g_tp = {
    .n_threads = 1,
    .shutdown = 0,
    .generation = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond_work = PTHREAD_COND_INITIALIZER,
    .cond_done = PTHREAD_COND_INITIALIZER,
};

static void *nemo_worker_loop(void *arg) {
    int tid = *(int *)arg;
    int my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&g_tp.mutex);
        while (g_tp.generation == my_gen && !g_tp.shutdown) {
            pthread_cond_wait(&g_tp.cond_work, &g_tp.mutex);
        }
        if (g_tp.shutdown) {
            pthread_mutex_unlock(&g_tp.mutex);
            return NULL;
        }
        my_gen = g_tp.generation;
        nemo_parallel_fn_t fn = g_tp.fn;
        void *a = g_tp.arg;
        int nt = g_tp.n_threads;
        pthread_mutex_unlock(&g_tp.mutex);

        fn(tid, nt, a);

        pthread_mutex_lock(&g_tp.mutex);
        if (++g_tp.n_done >= g_tp.n_threads - 1) {
            pthread_cond_signal(&g_tp.cond_done);
        }
        pthread_mutex_unlock(&g_tp.mutex);
    }
}

static void nemo_parallel_for(nemo_parallel_fn_t fn, void *arg) {
    if (g_tp.n_threads <= 1) {
        fn(0, 1, arg);
        return;
    }

    pthread_mutex_lock(&g_tp.mutex);
    g_tp.fn = fn;
    g_tp.arg = arg;
    g_tp.n_done = 0;
    g_tp.generation++;
    pthread_cond_broadcast(&g_tp.cond_work);
    pthread_mutex_unlock(&g_tp.mutex);

    fn(0, g_tp.n_threads, arg);

    pthread_mutex_lock(&g_tp.mutex);
    while (g_tp.n_done < g_tp.n_threads - 1) {
        pthread_cond_wait(&g_tp.cond_done, &g_tp.mutex);
    }
    pthread_mutex_unlock(&g_tp.mutex);
}

int nemo_get_num_cpus(void) {
#ifdef __APPLE__
    int n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.ncpu", &n, &len, NULL, 0) != 0) return 1;
    return n > 0 ? n : 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

void nemo_set_threads(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads > NEMO_MAX_THREADS) n_threads = NEMO_MAX_THREADS;
    if (n_threads == g_tp.n_threads) return;

    if (g_tp.n_threads > 1) {
        pthread_mutex_lock(&g_tp.mutex);
        g_tp.shutdown = 1;
        pthread_cond_broadcast(&g_tp.cond_work);
        pthread_mutex_unlock(&g_tp.mutex);
        for (int i = 0; i < g_tp.n_threads - 1; i++) {
            pthread_join(g_tp.threads[i], NULL);
        }
        pthread_mutex_lock(&g_tp.mutex);
        g_tp.shutdown = 0;
        g_tp.generation = 0;
        g_tp.fn = NULL;
        g_tp.arg = NULL;
        g_tp.n_done = 0;
        pthread_mutex_unlock(&g_tp.mutex);
    }

    g_tp.n_threads = n_threads;
    if (n_threads <= 1) return;

    for (int i = 0; i < n_threads - 1; i++) {
        g_tp.tids[i] = i + 1;
        if (pthread_create(&g_tp.threads[i], NULL, nemo_worker_loop, &g_tp.tids[i]) != 0) {
            g_tp.n_threads = i + 1;
            return;
        }
    }
}

int nemo_get_threads(void) {
    return g_tp.n_threads;
}

static int nemo_weight_is_bf16(const nemo_weight_t *w) {
    return w && w->dtype == NEMO_TENSOR_BF16 && w->bf16;
}

static int nemo_weight_is_q8(const nemo_weight_t *w) {
    return w && (w->dtype == NEMO_TENSOR_Q8 || w->dtype == NEMO_TENSOR_Q8P) &&
           w->q8 && w->q8_scales;
}

static int nemo_weight_q8_packed(const nemo_weight_t *w) {
    return w && (w->dtype == NEMO_TENSOR_Q8P || w->q8_packed);
}

static int nemo_weight_q8_stride(const nemo_weight_t *w, int fallback) {
    return (w && w->q8_stride > 0) ? (int)w->q8_stride : fallback;
}

static int nemo_weight_is_f32(const nemo_weight_t *w) {
    return w && w->dtype == NEMO_TENSOR_F32 && w->f32;
}

static float nemo_bf16_to_f32(uint16_t x) {
    uint32_t bits = (uint32_t)x << 16;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

static float nemo_dot_bf16_f32(const float *x, const uint16_t *w, int n) {
    return nemo_dot_bf16_f32_impl(x, w, n);
}

static float nemo_quantize_q8_symmetric_padded(const float *x, int8_t *x_q8,
                                               int n, int stride);

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

static int32_t nemo_dot_i8_i8_scalar(const int8_t *a, const int8_t *b, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

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
                                       const nemo_weight_t *w, int row,
                                       int stride, int n) {
    int32_t acc;
    if (nemo_weight_q8_packed(w)) {
        int q8_stride = nemo_weight_q8_stride(w, stride);
        acc = nemo_dot_i8_i8_packed_scalar(x_q8, w->q8, row, q8_stride);
    } else {
        acc = nemo_dot_i8_i8_scalar(x_q8, w->q8 + (size_t)row * stride, n);
    }
    return (float)acc * x_scale * w->q8_scales[row];
}

static float nemo_q8_dot_row(const float *x, const nemo_weight_t *w, int row, int stride, int n) {
    int8_t stack_q8[NEMO_Q8_STACK_MAX];
    int q8_stride = nemo_weight_q8_packed(w) ? nemo_weight_q8_stride(w, stride) : n;
    int8_t *x_q8 = nemo_q8_tmp_alloc(q8_stride, stack_q8);
    if (!x_q8) return 0.0f;
    float x_scale = nemo_quantize_q8_symmetric_padded(x, x_q8, n, q8_stride);
    float v = nemo_q8_dot_quantized_row(x_q8, x_scale, w, row, stride, n);
    nemo_q8_tmp_free(x_q8, stack_q8);
    return v;
}

static float nemo_weight_dot_row(const float *x, const nemo_weight_t *w, int row, int stride, int n) {
    if (nemo_weight_is_bf16(w)) {
        return nemo_dot_bf16_f32(x, w->bf16 + (size_t)row * stride, n);
    }
    if (nemo_weight_is_q8(w)) {
        return nemo_q8_dot_row(x, w, row, stride, n);
    }
    return nemo_dot_f32_impl(x, w->f32 + (size_t)row * stride, n);
}

static float nemo_weight_value_row(const nemo_weight_t *w, int row, int stride, int col) {
    size_t idx = (size_t)row * stride + col;
    if (nemo_weight_is_bf16(w)) return nemo_bf16_to_f32(w->bf16[idx]);
    if (nemo_weight_is_q8(w)) {
        int8_t qv = nemo_weight_q8_packed(w) ?
                    nemo_q8p_value(w->q8, row, nemo_weight_q8_stride(w, stride), col) :
                    w->q8[idx];
        return (float)qv * w->q8_scales[row];
    }
    return w->f32[idx];
}

typedef struct {
    float *y;
    const float *x;
    const float *w;
    const float *b;
    int rows;
    int in_dim;
    int out_dim;
} nemo_linear_task_t;

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
    const float *x;
    const uint16_t *w;
    const float *b;
    int rows;
    int in_dim;
    int out_dim;
} nemo_bf16_linear_task_t;

typedef struct {
    float *y;
    const float *x;
    const nemo_weight_t *w;
    const float *b;
    int rows;
    int in_dim;
    int out_dim;
} nemo_q8_linear_task_t;

static void nemo_linear_worker(int tid, int n_threads, void *arg) {
    nemo_linear_task_t *t = (nemo_linear_task_t *)arg;
    int total = t->rows * t->out_dim;
    int start = (total * tid) / n_threads;
    int end = (total * (tid + 1)) / n_threads;
    for (int idx = start; idx < end; idx++) {
        int r = idx / t->out_dim;
        int o = idx - r * t->out_dim;
        const float *xr = t->x + (size_t)r * t->in_dim;
        const float *wr = t->w + (size_t)o * t->in_dim;
        t->y[(size_t)r * t->out_dim + o] =
            nemo_dot_f32_impl(xr, wr, t->in_dim) + (t->b ? t->b[o] : 0.0f);
    }
}

static void nemo_linear_weight_worker(int tid, int n_threads, void *arg) {
    nemo_linear_weight_task_t *t = (nemo_linear_weight_task_t *)arg;
    int total = t->rows * t->out_dim;
    int start = (total * tid) / n_threads;
    int end = (total * (tid + 1)) / n_threads;
    for (int idx = start; idx < end; idx++) {
        int r = idx / t->out_dim;
        int o = idx - r * t->out_dim;
        const float *xr = t->x + (size_t)r * t->in_dim;
        t->y[(size_t)r * t->out_dim + o] =
            nemo_weight_dot_row(xr, t->w, o, t->in_dim, t->in_dim) + (t->b ? t->b[o] : 0.0f);
    }
}

static void nemo_bf16_linear_worker(int tid, int n_threads, void *arg) {
    nemo_bf16_linear_task_t *t = (nemo_bf16_linear_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    if (start >= end) return;
    for (int r = 0; r < t->rows; r++) {
        nemo_bf16_matvec_fused_impl(t->y + (size_t)r * t->out_dim + start,
                                    t->x + (size_t)r * t->in_dim,
                                    t->w + (size_t)start * t->in_dim,
                                    t->b ? t->b + start : NULL,
                                    t->in_dim, end - start);
    }
}

static void nemo_q8_linear_worker(int tid, int n_threads, void *arg) {
    nemo_q8_linear_task_t *t = (nemo_q8_linear_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    if (start >= end) return;
    int q8_stride = nemo_weight_q8_packed(t->w) ? nemo_weight_q8_stride(t->w, t->in_dim) : t->in_dim;
    int8_t stack_q8[NEMO_Q8_STACK_MAX];
    int8_t *x_q8 = nemo_q8_tmp_alloc(q8_stride, stack_q8);
    if (!x_q8) return;
    for (int r = 0; r < t->rows; r++) {
        const float *xr = t->x + (size_t)r * t->in_dim;
        float x_scale = nemo_quantize_q8_symmetric_padded(xr, x_q8, t->in_dim, q8_stride);
        if (nemo_weight_q8_packed(t->w)) {
            nemo_q8p_matvec_fused_impl(t->y + (size_t)r * t->out_dim + start,
                                       x_q8, x_scale, t->w->q8, t->w->q8_scales,
                                       t->b ? t->b + start : NULL,
                                       q8_stride, start, end - start);
        } else {
            nemo_q8_matvec_fused_impl(t->y + (size_t)r * t->out_dim + start,
                                      x_q8, x_scale,
                                      t->w->q8 + (size_t)start * t->in_dim,
                                      t->w->q8_scales + start,
                                      t->b ? t->b + start : NULL,
                                      t->in_dim, end - start);
        }
    }
    nemo_q8_tmp_free(x_q8, stack_q8);
}

typedef struct {
    float *y0;
    float *y1;
    float *y2;
    const float *x;
    const float *w0;
    const float *w1;
    const float *w2;
    int rows;
    int in_dim;
    int out_dim;
} nemo_linear3_task_t;

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
    const float *x;
    const uint16_t *w0;
    const uint16_t *w1;
    const uint16_t *w2;
    int rows;
    int in_dim;
    int out_dim;
} nemo_bf16_linear3_task_t;

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
} nemo_q8_linear3_task_t;

static void nemo_linear3_worker(int tid, int n_threads, void *arg) {
    nemo_linear3_task_t *t = (nemo_linear3_task_t *)arg;
    int total = t->rows * t->out_dim;
    int start = (total * tid) / n_threads;
    int end = (total * (tid + 1)) / n_threads;
    for (int idx = start; idx < end; idx++) {
        int r = idx / t->out_dim;
        int o = idx - r * t->out_dim;
        const float *xr = t->x + (size_t)r * t->in_dim;
        t->y0[(size_t)r * t->out_dim + o] =
            nemo_dot_f32_impl(xr, t->w0 + (size_t)o * t->in_dim, t->in_dim);
        t->y1[(size_t)r * t->out_dim + o] =
            nemo_dot_f32_impl(xr, t->w1 + (size_t)o * t->in_dim, t->in_dim);
        t->y2[(size_t)r * t->out_dim + o] =
            nemo_dot_f32_impl(xr, t->w2 + (size_t)o * t->in_dim, t->in_dim);
    }
}

static void nemo_linear3_weight_worker(int tid, int n_threads, void *arg) {
    nemo_linear3_weight_task_t *t = (nemo_linear3_weight_task_t *)arg;
    int total = t->rows * t->out_dim;
    int start = (total * tid) / n_threads;
    int end = (total * (tid + 1)) / n_threads;
    for (int idx = start; idx < end; idx++) {
        int r = idx / t->out_dim;
        int o = idx - r * t->out_dim;
        const float *xr = t->x + (size_t)r * t->in_dim;
        t->y0[(size_t)r * t->out_dim + o] =
            nemo_weight_dot_row(xr, t->w0, o, t->in_dim, t->in_dim);
        t->y1[(size_t)r * t->out_dim + o] =
            nemo_weight_dot_row(xr, t->w1, o, t->in_dim, t->in_dim);
        t->y2[(size_t)r * t->out_dim + o] =
            nemo_weight_dot_row(xr, t->w2, o, t->in_dim, t->in_dim);
    }
}

static void nemo_bf16_linear3_worker(int tid, int n_threads, void *arg) {
    nemo_bf16_linear3_task_t *t = (nemo_bf16_linear3_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    if (start >= end) return;
    for (int r = 0; r < t->rows; r++) {
        const float *xr = t->x + (size_t)r * t->in_dim;
        nemo_bf16_matvec_fused_impl(t->y0 + (size_t)r * t->out_dim + start,
                                    xr, t->w0 + (size_t)start * t->in_dim,
                                    NULL, t->in_dim, end - start);
        nemo_bf16_matvec_fused_impl(t->y1 + (size_t)r * t->out_dim + start,
                                    xr, t->w1 + (size_t)start * t->in_dim,
                                    NULL, t->in_dim, end - start);
        nemo_bf16_matvec_fused_impl(t->y2 + (size_t)r * t->out_dim + start,
                                    xr, t->w2 + (size_t)start * t->in_dim,
                                    NULL, t->in_dim, end - start);
    }
}

static void nemo_q8_linear3_worker(int tid, int n_threads, void *arg) {
    nemo_q8_linear3_task_t *t = (nemo_q8_linear3_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    if (start >= end) return;
    int q8_stride0 = nemo_weight_q8_packed(t->w0) ? nemo_weight_q8_stride(t->w0, t->in_dim) : t->in_dim;
    int q8_stride1 = nemo_weight_q8_packed(t->w1) ? nemo_weight_q8_stride(t->w1, t->in_dim) : t->in_dim;
    int q8_stride2 = nemo_weight_q8_packed(t->w2) ? nemo_weight_q8_stride(t->w2, t->in_dim) : t->in_dim;
    int q8_stride = q8_stride0;
    if (q8_stride1 > q8_stride) q8_stride = q8_stride1;
    if (q8_stride2 > q8_stride) q8_stride = q8_stride2;
    int8_t stack_q8[NEMO_Q8_STACK_MAX];
    int8_t *x_q8 = nemo_q8_tmp_alloc(q8_stride, stack_q8);
    if (!x_q8) return;
    for (int r = 0; r < t->rows; r++) {
        const float *xr = t->x + (size_t)r * t->in_dim;
        float x_scale = nemo_quantize_q8_symmetric_padded(xr, x_q8, t->in_dim, q8_stride);
        if (nemo_weight_q8_packed(t->w0)) {
            nemo_q8p_matvec_fused_impl(t->y0 + (size_t)r * t->out_dim + start,
                                       x_q8, x_scale, t->w0->q8, t->w0->q8_scales,
                                       NULL, q8_stride0, start, end - start);
        } else {
            nemo_q8_matvec_fused_impl(t->y0 + (size_t)r * t->out_dim + start,
                                      x_q8, x_scale, t->w0->q8 + (size_t)start * t->in_dim,
                                      t->w0->q8_scales + start, NULL, t->in_dim, end - start);
        }
        if (nemo_weight_q8_packed(t->w1)) {
            nemo_q8p_matvec_fused_impl(t->y1 + (size_t)r * t->out_dim + start,
                                       x_q8, x_scale, t->w1->q8, t->w1->q8_scales,
                                       NULL, q8_stride1, start, end - start);
        } else {
            nemo_q8_matvec_fused_impl(t->y1 + (size_t)r * t->out_dim + start,
                                      x_q8, x_scale, t->w1->q8 + (size_t)start * t->in_dim,
                                      t->w1->q8_scales + start, NULL, t->in_dim, end - start);
        }
        if (nemo_weight_q8_packed(t->w2)) {
            nemo_q8p_matvec_fused_impl(t->y2 + (size_t)r * t->out_dim + start,
                                       x_q8, x_scale, t->w2->q8, t->w2->q8_scales,
                                       NULL, q8_stride2, start, end - start);
        } else {
            nemo_q8_matvec_fused_impl(t->y2 + (size_t)r * t->out_dim + start,
                                      x_q8, x_scale, t->w2->q8 + (size_t)start * t->in_dim,
                                      t->w2->q8_scales + start, NULL, t->in_dim, end - start);
        }
    }
    nemo_q8_tmp_free(x_q8, stack_q8);
}

typedef struct {
    float *y;
    const float *x;
    const float *w;
    const float *b;
    int in_dim;
    int out_dim;
} nemo_matvec_task_t;

static void nemo_matvec_worker(int tid, int n_threads, void *arg) {
    nemo_matvec_task_t *t = (nemo_matvec_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    for (int o = start; o < end; o++) {
        const float *wr = t->w + (size_t)o * t->in_dim;
        t->y[o] = nemo_dot_f32_impl(t->x, wr, t->in_dim) + (t->b ? t->b[o] : 0.0f);
    }
}

typedef struct {
    const float *x;
    const float *w;
    const float *b;
    int in_dim;
    int out_dim;
    int best[NEMO_MAX_THREADS];
    float best_val[NEMO_MAX_THREADS];
} nemo_argmax_task_t;

static void nemo_argmax_worker(int tid, int n_threads, void *arg) {
    nemo_argmax_task_t *t = (nemo_argmax_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    int best = start;
    float best_val = -3.4028234663852886e38f;
    for (int o = start; o < end; o++) {
        const float *wr = t->w + (size_t)o * t->in_dim;
        float v = nemo_dot_f32_impl(t->x, wr, t->in_dim) + (t->b ? t->b[o] : 0.0f);
        if (v > best_val) {
            best_val = v;
            best = o;
        }
    }
    t->best[tid] = best;
    t->best_val[tid] = best_val;
}

typedef struct {
    const float *x;
    const nemo_weight_t *w;
    const float *b;
    int in_dim;
    int out_dim;
    int best[NEMO_MAX_THREADS];
    float best_val[NEMO_MAX_THREADS];
} nemo_argmax_weight_task_t;

static void nemo_argmax_weight_worker(int tid, int n_threads, void *arg) {
    nemo_argmax_weight_task_t *t = (nemo_argmax_weight_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    if (nemo_weight_is_bf16(t->w)) {
        t->best[tid] = nemo_argmax_bf16_range_impl(t->x, t->w->bf16, t->b,
                                                   t->in_dim, start, end,
                                                   &t->best_val[tid]);
        return;
    }
    if (nemo_weight_is_q8(t->w)) {
        int8_t stack_q8[NEMO_Q8_STACK_MAX];
        int q8_stride = nemo_weight_q8_packed(t->w) ? nemo_weight_q8_stride(t->w, t->in_dim) : t->in_dim;
        int8_t *x_q8 = nemo_q8_tmp_alloc(q8_stride, stack_q8);
        if (!x_q8) {
            t->best[tid] = start;
            t->best_val[tid] = -3.4028234663852886e38f;
            return;
        }
        float x_scale = nemo_quantize_q8_symmetric_padded(t->x, x_q8, t->in_dim, q8_stride);
        if (nemo_weight_q8_packed(t->w)) {
            t->best[tid] = nemo_argmax_q8p_range_impl(x_q8, x_scale,
                                                      t->w->q8, t->w->q8_scales, t->b,
                                                      q8_stride, start, end,
                                                      &t->best_val[tid]);
        } else {
            t->best[tid] = nemo_argmax_q8_range_impl(x_q8, x_scale,
                                                     t->w->q8, t->w->q8_scales, t->b,
                                                     t->in_dim, start, end,
                                                     &t->best_val[tid]);
        }
        nemo_q8_tmp_free(x_q8, stack_q8);
        return;
    }
    int best = start;
    float best_val = -3.4028234663852886e38f;
    for (int o = start; o < end; o++) {
        float v = nemo_weight_dot_row(t->x, t->w, o, t->in_dim, t->in_dim) +
                  (t->b ? t->b[o] : 0.0f);
        if (v > best_val) {
            best_val = v;
            best = o;
        }
    }
    t->best[tid] = best;
    t->best_val[tid] = best_val;
}

typedef struct {
    float *y;
    const float *x;
    const float *w;
    const float *b;
    int rows;
    int in_dim;
    int prompt_dim;
    int prompt_id;
    int out_dim;
} nemo_prompt_task_t;

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
} nemo_prompt_weight_task_t;

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
    const float *w_ih;
    const float *w_hh;
    const float *b_ih;
    const float *b_hh;
    int dim;
    int out_dim;
} nemo_lstm_gates_task_t;

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
} nemo_lstm_gates_weight_task_t;

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

static void nemo_prompt_worker(int tid, int n_threads, void *arg) {
    nemo_prompt_task_t *t = (nemo_prompt_task_t *)arg;
    int total = t->rows * t->out_dim;
    int stride = t->in_dim + t->prompt_dim;
    int start = (total * tid) / n_threads;
    int end = (total * (tid + 1)) / n_threads;
    for (int idx = start; idx < end; idx++) {
        int r = idx / t->out_dim;
        int o = idx - r * t->out_dim;
        const float *xr = t->x + (size_t)r * t->in_dim;
        const float *wr = t->w + (size_t)o * stride;
        float v = (t->b ? t->b[o] : 0.0f) + wr[t->in_dim + t->prompt_id];
        v += nemo_dot_f32_impl(xr, wr, t->in_dim);
        t->y[(size_t)r * t->out_dim + o] = v > 0.0f ? v : 0.0f;
    }
}

static void nemo_prompt_weight_worker(int tid, int n_threads, void *arg) {
    nemo_prompt_weight_task_t *t = (nemo_prompt_weight_task_t *)arg;
    int total = t->rows * t->out_dim;
    int stride = t->in_dim + t->prompt_dim;
    int start = (total * tid) / n_threads;
    int end = (total * (tid + 1)) / n_threads;
    for (int idx = start; idx < end; idx++) {
        int r = idx / t->out_dim;
        int o = idx - r * t->out_dim;
        const float *xr = t->x + (size_t)r * t->in_dim;
        float v = (t->b ? t->b[o] : 0.0f) +
                  nemo_weight_value_row(t->w, o, stride, t->in_dim + t->prompt_id);
        v += nemo_weight_dot_row(xr, t->w, o, stride, t->in_dim);
        t->y[(size_t)r * t->out_dim + o] = v > 0.0f ? v : 0.0f;
    }
}

static void nemo_prompt_q8_worker(int tid, int n_threads, void *arg) {
    nemo_prompt_q8_task_t *t = (nemo_prompt_q8_task_t *)arg;
    int stride = t->in_dim + t->prompt_dim;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    if (start >= end) return;
    int q8_stride = nemo_weight_q8_packed(t->w) ? nemo_weight_q8_stride(t->w, stride) : t->in_dim;
    int8_t stack_q8[NEMO_Q8_STACK_MAX];
    int8_t *x_q8 = nemo_q8_tmp_alloc(q8_stride, stack_q8);
    if (!x_q8) return;
    for (int r = 0; r < t->rows; r++) {
        const float *xr = t->x + (size_t)r * t->in_dim;
        float x_scale = nemo_quantize_q8_symmetric_padded(xr, x_q8, t->in_dim, q8_stride);
        for (int o = start; o < end; o++) {
            float v = (t->b ? t->b[o] : 0.0f) +
                      nemo_weight_value_row(t->w, o, stride, t->in_dim + t->prompt_id);
            v += nemo_q8_dot_quantized_row(x_q8, x_scale, t->w, o, stride, t->in_dim);
            t->y[(size_t)r * t->out_dim + o] = v > 0.0f ? v : 0.0f;
        }
    }
    nemo_q8_tmp_free(x_q8, stack_q8);
}

static void nemo_lstm_gates_worker(int tid, int n_threads, void *arg) {
    nemo_lstm_gates_task_t *t = (nemo_lstm_gates_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    for (int o = start; o < end; o++) {
        float v = 0.0f;
        if (t->b_ih) v += t->b_ih[o];
        if (t->b_hh) v += t->b_hh[o];
        v += nemo_dot_f32_impl(t->x, t->w_ih + (size_t)o * t->dim, t->dim);
        v += nemo_dot_f32_impl(t->h, t->w_hh + (size_t)o * t->dim, t->dim);
        t->y[o] = v;
    }
}

static void nemo_lstm_gates_weight_worker(int tid, int n_threads, void *arg) {
    nemo_lstm_gates_weight_task_t *t = (nemo_lstm_gates_weight_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    for (int o = start; o < end; o++) {
        float v = 0.0f;
        if (t->b_ih) v += t->b_ih[o];
        if (t->b_hh) v += t->b_hh[o];
        v += nemo_weight_dot_row(t->x, t->w_ih, o, t->dim, t->dim);
        v += nemo_weight_dot_row(t->h, t->w_hh, o, t->dim, t->dim);
        t->y[o] = v;
    }
}

static void nemo_lstm_gates_q8_worker(int tid, int n_threads, void *arg) {
    nemo_lstm_gates_q8_task_t *t = (nemo_lstm_gates_q8_task_t *)arg;
    int start = (t->out_dim * tid) / n_threads;
    int end = (t->out_dim * (tid + 1)) / n_threads;
    if (start >= end) return;
    int x_q8_stride = nemo_weight_q8_packed(t->w_ih) ? nemo_weight_q8_stride(t->w_ih, t->dim) : t->dim;
    int h_q8_stride = nemo_weight_q8_packed(t->w_hh) ? nemo_weight_q8_stride(t->w_hh, t->dim) : t->dim;
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
        v += nemo_q8_dot_quantized_row(x_q8, x_scale, t->w_ih, o, t->dim, t->dim);
        v += nemo_q8_dot_quantized_row(h_q8, h_scale, t->w_hh, o, t->dim, t->dim);
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
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

void nemo_linear(float *y, const float *x, const float *w, const float *b,
                 int rows, int in_dim, int out_dim) {
    long long work = (long long)rows * (long long)in_dim * (long long)out_dim;
#ifdef USE_BLAS
    if (rows >= NEMO_BLAS_ROWS_MIN && work >= NEMO_PARALLEL_WORK_MIN) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    rows, out_dim, in_dim,
                    1.0f, x, in_dim, w, in_dim,
                    0.0f, y, out_dim);
        if (b) {
            for (int r = 0; r < rows; r++) {
                float *yr = y + (size_t)r * out_dim;
                for (int o = 0; o < out_dim; o++) yr[o] += b[o];
            }
        }
        return;
    }
#endif
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        nemo_linear_task_t task = {y, x, w, b, rows, in_dim, out_dim};
        nemo_parallel_for(nemo_linear_worker, &task);
        return;
    }
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * in_dim;
        float *yr = y + (size_t)r * out_dim;
        nemo_matvec_f32_impl(yr, xr, w, b, in_dim, out_dim);
    }
}

void nemo_linear_nobias(float *y, const float *x, const float *w,
                        int rows, int in_dim, int out_dim) {
    nemo_linear(y, x, w, NULL, rows, in_dim, out_dim);
}

void nemo_linear_weight(float *y, const float *x, const nemo_weight_t *w, const float *b,
                        int rows, int in_dim, int out_dim) {
    if (nemo_weight_is_f32(w)) {
        nemo_linear(y, x, w->f32, b, rows, in_dim, out_dim);
        return;
    }
    long long work = (long long)rows * (long long)in_dim * (long long)out_dim;
    if (nemo_weight_is_bf16(w)) {
        nemo_bf16_linear_task_t task = {y, x, w->bf16, b, rows, in_dim, out_dim};
        if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
            nemo_parallel_for(nemo_bf16_linear_worker, &task);
            return;
        }
        nemo_bf16_linear_worker(0, 1, &task);
        return;
    }
    if (nemo_weight_is_q8(w)) {
        nemo_q8_linear_task_t task = {y, x, w, b, rows, in_dim, out_dim};
        if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
            nemo_parallel_for(nemo_q8_linear_worker, &task);
            return;
        }
        nemo_q8_linear_worker(0, 1, &task);
        return;
    }
    nemo_linear_weight_task_t task = {y, x, w, b, rows, in_dim, out_dim};
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        nemo_parallel_for(nemo_linear_weight_worker, &task);
        return;
    }
    nemo_linear_weight_worker(0, 1, &task);
}

void nemo_linear_nobias_weight(float *y, const float *x, const nemo_weight_t *w,
                               int rows, int in_dim, int out_dim) {
    nemo_linear_weight(y, x, w, NULL, rows, in_dim, out_dim);
}

void nemo_linear3_nobias(float *y0, float *y1, float *y2, const float *x,
                         const float *w0, const float *w1, const float *w2,
                         int rows, int in_dim, int out_dim) {
    long long work = (long long)rows * (long long)in_dim * (long long)out_dim * 3LL;
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        nemo_linear3_task_t task = {y0, y1, y2, x, w0, w1, w2, rows, in_dim, out_dim};
        nemo_parallel_for(nemo_linear3_worker, &task);
        return;
    }
    nemo_linear_nobias(y0, x, w0, rows, in_dim, out_dim);
    nemo_linear_nobias(y1, x, w1, rows, in_dim, out_dim);
    nemo_linear_nobias(y2, x, w2, rows, in_dim, out_dim);
}

void nemo_linear3_nobias_weight(float *y0, float *y1, float *y2, const float *x,
                                const nemo_weight_t *w0, const nemo_weight_t *w1,
                                const nemo_weight_t *w2,
                                int rows, int in_dim, int out_dim) {
    if (nemo_weight_is_f32(w0) && nemo_weight_is_f32(w1) && nemo_weight_is_f32(w2)) {
        nemo_linear3_nobias(y0, y1, y2, x, w0->f32, w1->f32, w2->f32,
                            rows, in_dim, out_dim);
        return;
    }
    long long work = (long long)rows * (long long)in_dim * (long long)out_dim * 3LL;
    if (nemo_weight_is_bf16(w0) && nemo_weight_is_bf16(w1) && nemo_weight_is_bf16(w2)) {
        nemo_bf16_linear3_task_t task = {y0, y1, y2, x,
                                         w0->bf16, w1->bf16, w2->bf16,
                                         rows, in_dim, out_dim};
        if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
            nemo_parallel_for(nemo_bf16_linear3_worker, &task);
            return;
        }
        nemo_bf16_linear3_worker(0, 1, &task);
        return;
    }
    if (nemo_weight_is_q8(w0) && nemo_weight_is_q8(w1) && nemo_weight_is_q8(w2)) {
        nemo_q8_linear3_task_t task = {y0, y1, y2, x, w0, w1, w2, rows, in_dim, out_dim};
        if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
            nemo_parallel_for(nemo_q8_linear3_worker, &task);
            return;
        }
        nemo_q8_linear3_worker(0, 1, &task);
        return;
    }
    nemo_linear3_weight_task_t task = {y0, y1, y2, x, w0, w1, w2, rows, in_dim, out_dim};
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        nemo_parallel_for(nemo_linear3_weight_worker, &task);
        return;
    }
    nemo_linear3_weight_worker(0, 1, &task);
}

void nemo_prompt_linear_relu(float *y, const float *x, const float *w, const float *b,
                             int rows, int in_dim, int prompt_dim, int prompt_id,
                             int out_dim) {
    if (prompt_id < 0 || prompt_id >= prompt_dim) prompt_id = prompt_dim - 1;
    long long work = (long long)rows * (long long)in_dim * (long long)out_dim;
    nemo_prompt_task_t task = {y, x, w, b, rows, in_dim, prompt_dim, prompt_id, out_dim};
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        nemo_parallel_for(nemo_prompt_worker, &task);
        return;
    }
    nemo_prompt_worker(0, 1, &task);
}

void nemo_prompt_linear_relu_weight(float *y, const float *x, const nemo_weight_t *w,
                                    const float *b, int rows, int in_dim,
                                    int prompt_dim, int prompt_id, int out_dim) {
    if (nemo_weight_is_f32(w)) {
        nemo_prompt_linear_relu(y, x, w->f32, b, rows, in_dim, prompt_dim, prompt_id, out_dim);
        return;
    }
    if (prompt_id < 0 || prompt_id >= prompt_dim) prompt_id = prompt_dim - 1;
    long long work = (long long)rows * (long long)in_dim * (long long)out_dim;
    if (nemo_weight_is_q8(w)) {
        nemo_prompt_q8_task_t task = {y, x, w, b, rows, in_dim, prompt_dim, prompt_id, out_dim};
        if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
            nemo_parallel_for(nemo_prompt_q8_worker, &task);
            return;
        }
        nemo_prompt_q8_worker(0, 1, &task);
        return;
    }
    nemo_prompt_weight_task_t task = {y, x, w, b, rows, in_dim, prompt_dim, prompt_id, out_dim};
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        nemo_parallel_for(nemo_prompt_weight_worker, &task);
        return;
    }
    nemo_prompt_weight_worker(0, 1, &task);
}

void nemo_lstm_gates_f32(float *y, const float *x, const float *h,
                         const float *w_ih, const float *w_hh,
                         const float *b_ih, const float *b_hh,
                         int dim, int out_dim) {
    long long work = (long long)dim * (long long)out_dim * 2LL;
    nemo_lstm_gates_task_t task = {y, x, h, w_ih, w_hh, b_ih, b_hh, dim, out_dim};
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        nemo_parallel_for(nemo_lstm_gates_worker, &task);
        return;
    }
    nemo_lstm_gates_worker(0, 1, &task);
}

void nemo_lstm_gates_weight(float *y, const float *x, const float *h,
                            const nemo_weight_t *w_ih, const nemo_weight_t *w_hh,
                            const float *b_ih, const float *b_hh,
                            int dim, int out_dim) {
    if (nemo_weight_is_f32(w_ih) && nemo_weight_is_f32(w_hh)) {
        nemo_lstm_gates_f32(y, x, h, w_ih->f32, w_hh->f32, b_ih, b_hh, dim, out_dim);
        return;
    }
    long long work = (long long)dim * (long long)out_dim * 2LL;
    if (nemo_weight_is_q8(w_ih) && nemo_weight_is_q8(w_hh)) {
        nemo_lstm_gates_q8_task_t task = {y, x, h, w_ih, w_hh, b_ih, b_hh, dim, out_dim};
        if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
            nemo_parallel_for(nemo_lstm_gates_q8_worker, &task);
            return;
        }
        nemo_lstm_gates_q8_worker(0, 1, &task);
        return;
    }
    nemo_lstm_gates_weight_task_t task = {y, x, h, w_ih, w_hh, b_ih, b_hh, dim, out_dim};
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        nemo_parallel_for(nemo_lstm_gates_weight_worker, &task);
        return;
    }
    nemo_lstm_gates_weight_worker(0, 1, &task);
}

static int conv_out_len(int n, int left, int right, int k, int stride) {
    return (n + left + right - k) / stride + 1;
}

void nemo_conv2d(float *out, const float *in, const float *w, const float *b,
                 int c_in, int c_out, int t_in, int f_in,
                 int k, int stride, int left, int right, int groups) {
    int t_out = conv_out_len(t_in, left, right, k, stride);
    int f_out = conv_out_len(f_in, left, right, k, stride);
    int in_per_group = c_in / groups;
    int out_per_group = c_out / groups;
    for (int oc = 0; oc < c_out; oc++) {
        int g = oc / out_per_group;
        int ic0 = g * in_per_group;
        for (int ot = 0; ot < t_out; ot++) {
            for (int of = 0; of < f_out; of++) {
                float sum = b ? b[oc] : 0.0f;
                for (int icg = 0; icg < in_per_group; icg++) {
                    int ic = ic0 + icg;
                    int w_ic = (groups == c_in && in_per_group == 1) ? 0 : icg;
                    for (int kt = 0; kt < k; kt++) {
                        int it = ot * stride + kt - left;
                        if (it < 0 || it >= t_in) continue;
                        for (int kf = 0; kf < k; kf++) {
                            int iff = of * stride + kf - left;
                            if (iff < 0 || iff >= f_in) continue;
                            size_t iidx = ((size_t)ic * t_in + it) * f_in + iff;
                            size_t widx = (((size_t)oc * in_per_group + w_ic) * k + kt) * k + kf;
                            sum += in[iidx] * w[widx];
                        }
                    }
                }
                out[((size_t)oc * t_out + ot) * f_out + of] = sum;
            }
        }
    }
}

void nemo_conv1d_depthwise_causal(float *out, const float *in, const float *w,
                                  int t, int dim, int k, int left) {
    for (int tt = 0; tt < t; tt++) {
        for (int c = 0; c < dim; c++) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; kk++) {
                int src = tt + kk - left;
                if (src >= 0 && src < t) {
                    sum += in[(size_t)src * dim + c] * w[(size_t)c * k + kk];
                }
            }
            out[(size_t)tt * dim + c] = sum;
        }
    }
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

void nemo_matvec_f32(float *y, const float *x, const float *w, const float *b,
                     int in_dim, int out_dim) {
    long long work = (long long)in_dim * (long long)out_dim;
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        nemo_matvec_task_t task = {y, x, w, b, in_dim, out_dim};
        nemo_parallel_for(nemo_matvec_worker, &task);
        return;
    }
    nemo_matvec_f32_impl(y, x, w, b, in_dim, out_dim);
}

int nemo_argmax_matvec_f32(const float *x, const float *w, const float *b,
                           int in_dim, int out_dim, float *best_val_out) {
    long long work = (long long)in_dim * (long long)out_dim;
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        nemo_argmax_task_t task;
        memset(&task, 0, sizeof(task));
        task.x = x;
        task.w = w;
        task.b = b;
        task.in_dim = in_dim;
        task.out_dim = out_dim;
        nemo_parallel_for(nemo_argmax_worker, &task);
        int best = 0;
        float best_val = -3.4028234663852886e38f;
        for (int i = 0; i < g_tp.n_threads; i++) {
            if (task.best_val[i] > best_val) {
                best_val = task.best_val[i];
                best = task.best[i];
            }
        }
        if (best_val_out) *best_val_out = best_val;
        return best;
    }
    return nemo_argmax_matvec_f32_impl(x, w, b, in_dim, out_dim, best_val_out);
}

int nemo_argmax_matvec_weight(const float *x, const nemo_weight_t *w, const float *b,
                              int in_dim, int out_dim, float *best_val_out) {
    if (nemo_weight_is_f32(w)) {
        return nemo_argmax_matvec_f32(x, w->f32, b, in_dim, out_dim, best_val_out);
    }
    long long work = (long long)in_dim * (long long)out_dim;
    nemo_argmax_weight_task_t task;
    memset(&task, 0, sizeof(task));
    task.x = x;
    task.w = w;
    task.b = b;
    task.in_dim = in_dim;
    task.out_dim = out_dim;
    if (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) {
        nemo_parallel_for(nemo_argmax_weight_worker, &task);
    } else {
        nemo_argmax_weight_worker(0, 1, &task);
    }
    int n_threads = (g_tp.n_threads > 1 && work >= NEMO_PARALLEL_WORK_MIN) ? g_tp.n_threads : 1;
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

void nemo_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    float inv = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
    for (int i = 0; i < n; i++) x[i] *= inv;
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

void nemo_relu(float *x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}
