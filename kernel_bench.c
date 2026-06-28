#include "nemotron_asr_kernels.h"
#include "nemotron_asr_kernels_impl.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/time.h>
#endif

static uint32_t rng_state = 0x2468ace1u;
static volatile float sink_f32;

static uint32_t next_u32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static int8_t next_i8(void) {
    return (int8_t)((int)(next_u32() % 255u) - 127);
}

static float next_f32(float scale) {
    int v = (int)(next_u32() % 2001u) - 1000;
    return (float)v * scale;
}

static uint16_t f32_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static double now_ms(void) {
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

static float checksum(const float *x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * (float)((i % 13) + 1);
    return s;
}

static int align_int(int v, int a) {
    return (v + a - 1) & ~(a - 1);
}

static void pack_q8p(int8_t *packed, const int8_t *w, int in_dim, int out_dim) {
    int stride = align_int(in_dim, 16);
    int packed_rows = align_int(out_dim, 4);
    memset(packed, 0, (size_t)packed_rows * (size_t)stride);
    for (int o = 0; o < out_dim; o++) {
        int tile = o >> 2;
        int lane = o & 3;
        for (int k = 0; k < in_dim; k++) {
            size_t dst = (size_t)tile * 4u * (size_t)stride +
                         (size_t)(k >> 4) * 64u +
                         (size_t)lane * 16u + (size_t)(k & 15);
            packed[dst] = w[(size_t)o * in_dim + k];
        }
    }
}

static void print_rate(const char *name, const char *kind, const char *backend,
                       int in_dim, int out_dim, int iters, double ms,
                       double speedup) {
    double ops = 2.0 * (double)in_dim * (double)out_dim * (double)iters;
    double gops = ops / (ms * 1.0e6);
    printf("%-16s %-6s %-7s %5d -> %-5d  %6d iters  %8.3f ms  %8.2f GOPS  %6.2fx\n",
           name, kind, backend, in_dim, out_dim, iters, ms, gops, speedup);
}

static int bench_q8(const char *name, int in_dim, int out_dim, int iters) {
    int stride = align_int(in_dim, 16);
    int packed_rows = align_int(out_dim, 4);
    int8_t *x = (int8_t *)malloc((size_t)in_dim);
    int8_t *xpad = (int8_t *)calloc((size_t)stride, 1);
    int8_t *w = (int8_t *)malloc((size_t)in_dim * (size_t)out_dim);
    int8_t *wp = (int8_t *)malloc((size_t)packed_rows * (size_t)stride);
    float *scales = (float *)malloc((size_t)out_dim * sizeof(float));
    float *bias = (float *)malloc((size_t)out_dim * sizeof(float));
    float *y = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!x || !xpad || !w || !wp || !scales || !bias || !y) {
        fprintf(stderr, "allocation failed in q8 bench\n");
        free(x);
        free(xpad);
        free(w);
        free(wp);
        free(scales);
        free(bias);
        free(y);
        return -1;
    }
    for (int i = 0; i < in_dim; i++) x[i] = next_i8();
    memcpy(xpad, x, (size_t)in_dim);
    for (int i = 0; i < in_dim * out_dim; i++) w[i] = next_i8();
    pack_q8p(wp, w, in_dim, out_dim);
    for (int o = 0; o < out_dim; o++) {
        scales[o] = 0.0002f + 0.00001f * (float)(o % 11);
        bias[o] = next_f32(0.001f);
    }
    float x_scale = 0.0007f;

    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        nemo_q8p_matvec_fused_generic(y, xpad, x_scale, wp, scales, bias, stride, 0, out_dim);
        sink_f32 += checksum(y, out_dim);
    }
    double t1 = now_ms();
    double generic_packed_ms = t1 - t0;

    t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        nemo_q8p_matvec_fused_impl(y, xpad, x_scale, wp, scales, bias, stride, 0, out_dim);
        sink_f32 += checksum(y, out_dim);
    }
    t1 = now_ms();
    double native_packed_ms = t1 - t0;
    print_rate(name, "q8p", "generic", in_dim, out_dim, iters, generic_packed_ms, 1.0);
    print_rate(name, "q8p", "native", in_dim, out_dim, iters, native_packed_ms,
               generic_packed_ms / native_packed_ms);

    free(x);
    free(xpad);
    free(w);
    free(wp);
    free(scales);
    free(bias);
    free(y);
    return 0;
}

static int bench_q8_argmax(const char *name, int in_dim, int out_dim, int iters) {
    int stride = align_int(in_dim, 16);
    int packed_rows = align_int(out_dim, 4);
    int8_t *x = (int8_t *)malloc((size_t)in_dim);
    int8_t *xpad = (int8_t *)calloc((size_t)stride, 1);
    int8_t *w = (int8_t *)malloc((size_t)in_dim * (size_t)out_dim);
    int8_t *wp = (int8_t *)malloc((size_t)packed_rows * (size_t)stride);
    float *scales = (float *)malloc((size_t)out_dim * sizeof(float));
    float *bias = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!x || !xpad || !w || !wp || !scales || !bias) {
        fprintf(stderr, "allocation failed in q8 argmax bench\n");
        free(x);
        free(xpad);
        free(w);
        free(wp);
        free(scales);
        free(bias);
        return -1;
    }
    for (int i = 0; i < in_dim; i++) x[i] = next_i8();
    memcpy(xpad, x, (size_t)in_dim);
    for (int i = 0; i < in_dim * out_dim; i++) w[i] = next_i8();
    pack_q8p(wp, w, in_dim, out_dim);
    for (int o = 0; o < out_dim; o++) {
        scales[o] = 0.0002f + 0.00001f * (float)(o % 11);
        bias[o] = next_f32(0.001f);
    }
    float x_scale = 0.0007f;
    float best_val = 0.0f;

    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        int best = nemo_argmax_q8p_range_generic(xpad, x_scale, wp, scales, bias,
                                                 stride, 0, out_dim, &best_val);
        sink_f32 += best_val + (float)best;
    }
    double t1 = now_ms();
    double generic_packed_ms = t1 - t0;

    t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        int best = nemo_argmax_q8p_range_impl(xpad, x_scale, wp, scales, bias,
                                              stride, 0, out_dim, &best_val);
        sink_f32 += best_val + (float)best;
    }
    t1 = now_ms();
    double native_packed_ms = t1 - t0;
    print_rate(name, "q8parg", "generic", in_dim, out_dim, iters, generic_packed_ms, 1.0);
    print_rate(name, "q8parg", "native", in_dim, out_dim, iters, native_packed_ms,
               generic_packed_ms / native_packed_ms);

    free(x);
    free(xpad);
    free(w);
    free(wp);
    free(scales);
    free(bias);
    return 0;
}

static int bench_q8_runtime(const char *name, int in_dim, int out_dim, int iters) {
    int stride = align_int(in_dim, 16);
    int packed_rows = align_int(out_dim, 4);
    float *x = (float *)malloc((size_t)in_dim * sizeof(float));
    int8_t *w = (int8_t *)malloc((size_t)in_dim * (size_t)out_dim);
    int8_t *wp = (int8_t *)malloc((size_t)packed_rows * (size_t)stride);
    float *scales = (float *)malloc((size_t)out_dim * sizeof(float));
    float *bias = (float *)malloc((size_t)out_dim * sizeof(float));
    float *y = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!x || !w || !wp || !scales || !bias || !y) {
        fprintf(stderr, "allocation failed in q8 runtime bench\n");
        free(x);
        free(w);
        free(wp);
        free(scales);
        free(bias);
        free(y);
        return -1;
    }
    for (int i = 0; i < in_dim; i++) x[i] = next_f32(0.002f);
    for (int i = 0; i < in_dim * out_dim; i++) w[i] = next_i8();
    pack_q8p(wp, w, in_dim, out_dim);
    for (int o = 0; o < out_dim; o++) {
        scales[o] = 0.0002f + 0.00001f * (float)(o % 11);
        bias[o] = next_f32(0.001f);
    }
    nemo_weight_t weight = {
        .q8 = wp,
        .q8_scales = scales,
        .q8_stride = (uint32_t)stride,
        .dtype = NEMO_TENSOR_Q8P,
    };

    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        nemo_linear_weight(y, x, &weight, bias, 1, in_dim, out_dim);
        sink_f32 += checksum(y, out_dim);
    }
    double t1 = now_ms();
    double linear_ms = t1 - t0;
    print_rate(name, "q8wrap", "runtime", in_dim, out_dim, iters, linear_ms, 1.0);

    float best_val = 0.0f;
    t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        int best = nemo_argmax_matvec_weight(x, &weight, bias, in_dim, out_dim, &best_val);
        sink_f32 += best_val + (float)best;
    }
    t1 = now_ms();
    double argmax_ms = t1 - t0;
    print_rate(name, "q8argw", "runtime", in_dim, out_dim, iters, argmax_ms, 1.0);

    free(x);
    free(w);
    free(wp);
    free(scales);
    free(bias);
    free(y);
    return 0;
}

static int bench_bf16(const char *name, int in_dim, int out_dim, int iters) {
    float *x = (float *)malloc((size_t)in_dim * sizeof(float));
    uint16_t *w = (uint16_t *)malloc((size_t)in_dim * (size_t)out_dim * sizeof(uint16_t));
    float *bias = (float *)malloc((size_t)out_dim * sizeof(float));
    float *y = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!x || !w || !bias || !y) {
        fprintf(stderr, "allocation failed in bf16 bench\n");
        free(x);
        free(w);
        free(bias);
        free(y);
        return -1;
    }
    for (int i = 0; i < in_dim; i++) x[i] = next_f32(0.002f);
    for (int i = 0; i < in_dim * out_dim; i++) w[i] = f32_to_bf16(next_f32(0.002f));
    for (int o = 0; o < out_dim; o++) bias[o] = next_f32(0.001f);

    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        nemo_bf16_matvec_fused_generic(y, x, w, bias, in_dim, out_dim);
        sink_f32 += checksum(y, out_dim);
    }
    double t1 = now_ms();
    double generic_ms = t1 - t0;

    t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        nemo_bf16_matvec_fused_impl(y, x, w, bias, in_dim, out_dim);
        sink_f32 += checksum(y, out_dim);
    }
    t1 = now_ms();
    double native_ms = t1 - t0;
    print_rate(name, "bf16", "generic", in_dim, out_dim, iters, generic_ms, 1.0);
    print_rate(name, "bf16", "native", in_dim, out_dim, iters, native_ms,
               generic_ms / native_ms);

    free(x);
    free(w);
    free(bias);
    free(y);
    return 0;
}

static int bench_bf16_argmax(const char *name, int in_dim, int out_dim, int iters) {
    float *x = (float *)malloc((size_t)in_dim * sizeof(float));
    uint16_t *w = (uint16_t *)malloc((size_t)in_dim * (size_t)out_dim * sizeof(uint16_t));
    float *bias = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!x || !w || !bias) {
        fprintf(stderr, "allocation failed in bf16 argmax bench\n");
        free(x);
        free(w);
        free(bias);
        return -1;
    }
    for (int i = 0; i < in_dim; i++) x[i] = next_f32(0.002f);
    for (int i = 0; i < in_dim * out_dim; i++) w[i] = f32_to_bf16(next_f32(0.002f));
    for (int o = 0; o < out_dim; o++) bias[o] = next_f32(0.001f);
    float best_val = 0.0f;

    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        int best = nemo_argmax_bf16_range_generic(x, w, bias, in_dim, 0, out_dim, &best_val);
        sink_f32 += best_val + (float)best;
    }
    double t1 = now_ms();
    double generic_ms = t1 - t0;

    t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        int best = nemo_argmax_bf16_range_impl(x, w, bias, in_dim, 0, out_dim, &best_val);
        sink_f32 += best_val + (float)best;
    }
    t1 = now_ms();
    double native_ms = t1 - t0;
    print_rate(name, "bfarg", "generic", in_dim, out_dim, iters, generic_ms, 1.0);
    print_rate(name, "bfarg", "native", in_dim, out_dim, iters, native_ms,
               generic_ms / native_ms);

    free(x);
    free(w);
    free(bias);
    return 0;
}

int main(void) {
    nemo_set_threads(1);

    struct bench_case {
        const char *name;
        int in_dim;
        int out_dim;
        int q8_iters;
        int bf16_iters;
    };
    static const struct bench_case cases[] = {
        {"ffn_expand", 1024, 4096, 160, 80},
        {"ffn_reduce", 4096, 1024, 200, 80},
        {"attn_proj", 1024, 1024, 800, 400},
        {"joint_vocab", 640, 13088, 120, 60},
    };

    printf("Nemotron kernel microbench (generic vs native backend)\n");
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (bench_q8(cases[i].name, cases[i].in_dim, cases[i].out_dim,
                     cases[i].q8_iters) != 0) {
            return 1;
        }
        if (bench_bf16(cases[i].name, cases[i].in_dim, cases[i].out_dim,
                       cases[i].bf16_iters) != 0) {
            return 1;
        }
        if (bench_q8_argmax(cases[i].name, cases[i].in_dim, cases[i].out_dim,
                            cases[i].q8_iters) != 0) {
            return 1;
        }
        if (bench_bf16_argmax(cases[i].name, cases[i].in_dim, cases[i].out_dim,
                              cases[i].bf16_iters) != 0) {
            return 1;
        }
        if (bench_q8_runtime(cases[i].name, cases[i].in_dim, cases[i].out_dim,
                             cases[i].q8_iters) != 0) {
            return 1;
        }
    }
    printf("checksum: %.6g\n", (double)sink_f32);
    return 0;
}
