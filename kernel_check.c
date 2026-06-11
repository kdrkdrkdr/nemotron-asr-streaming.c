#include "nemotron_asr_kernels_impl.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng_state = 0x13579bdfu;

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

static uint16_t f32_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static int nearly_equal(float a, float b, float atol, float rtol) {
    float d = fabsf(a - b);
    float m = fmaxf(fabsf(a), fabsf(b));
    return d <= atol + rtol * m;
}

static int check_q8p_matvec_case(int in_dim, int out_dim) {
    int stride = align_int(in_dim, 16);
    int packed_rows = align_int(out_dim, 4);
    int8_t *x = (int8_t *)malloc((size_t)in_dim);
    int8_t *xpad = (int8_t *)calloc((size_t)stride, 1);
    int8_t *w = (int8_t *)malloc((size_t)in_dim * (size_t)out_dim);
    int8_t *wp = (int8_t *)malloc((size_t)packed_rows * (size_t)stride);
    float *scales = (float *)malloc((size_t)out_dim * sizeof(float));
    float *bias = (float *)malloc((size_t)out_dim * sizeof(float));
    float *yref = (float *)malloc((size_t)out_dim * sizeof(float));
    float *ypg = (float *)malloc((size_t)out_dim * sizeof(float));
    float *ypi = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!x || !xpad || !w || !wp || !scales || !bias || !yref || !ypg || !ypi) {
        free(x);
        free(xpad);
        free(w);
        free(wp);
        free(scales);
        free(bias);
        free(yref);
        free(ypg);
        free(ypi);
        return -1;
    }

    for (int i = 0; i < in_dim; i++) x[i] = next_i8();
    memcpy(xpad, x, (size_t)in_dim);
    for (int i = 0; i < in_dim * out_dim; i++) w[i] = next_i8();
    pack_q8p(wp, w, in_dim, out_dim);
    for (int o = 0; o < out_dim; o++) {
        scales[o] = 0.0003f + 0.00001f * (float)(o % 17);
        bias[o] = next_f32(0.001f);
    }
    float x_scale = 0.0007f;

    for (int o = 0; o < out_dim; o++) {
        int32_t acc = 0;
        for (int i = 0; i < in_dim; i++) {
            acc += (int32_t)x[i] * (int32_t)w[(size_t)o * in_dim + i];
        }
        yref[o] = (float)acc * x_scale * scales[o] + bias[o];
    }

    nemo_q8p_matvec_fused_generic(ypg, xpad, x_scale, wp, scales, bias, stride, 0, out_dim);
    nemo_q8p_matvec_fused_impl(ypi, xpad, x_scale, wp, scales, bias, stride, 0, out_dim);
    for (int o = 0; o < out_dim; o++) {
        if (!nearly_equal(yref[o], ypg[o], 1e-5f, 1e-6f) ||
            !nearly_equal(ypg[o], ypi[o], 1e-5f, 1e-6f)) {
            fprintf(stderr, "q8p matvec mismatch in=%d out=%d o=%d ref=%g generic=%g impl=%g\n",
                    in_dim, out_dim, o, yref[o], ypg[o], ypi[o]);
            free(x);
            free(xpad);
            free(w);
            free(wp);
            free(scales);
            free(bias);
            free(yref);
            free(ypg);
            free(ypi);
            return -1;
        }
    }

    int start = out_dim > 3 ? 1 : 0;
    int end = out_dim > 3 ? out_dim - 1 : out_dim;
    int bref = start;
    float vref = -3.4028234663852886e38f;
    for (int o = start; o < end; o++) {
        if (yref[o] > vref) {
            vref = yref[o];
            bref = o;
        }
    }

    float vpg = 0.0f, vpi = 0.0f;
    int bpg = nemo_argmax_q8p_range_generic(xpad, x_scale, wp, scales, bias,
                                            stride, start, end, &vpg);
    int bpi = nemo_argmax_q8p_range_impl(xpad, x_scale, wp, scales, bias,
                                         stride, start, end, &vpi);
    if (bref != bpg || bpg != bpi ||
        !nearly_equal(vref, vpg, 1e-5f, 1e-6f) ||
        !nearly_equal(vpg, vpi, 1e-5f, 1e-6f)) {
        fprintf(stderr, "q8p argmax mismatch in=%d out=%d ref=(%d,%g) generic=(%d,%g) impl=(%d,%g)\n",
                in_dim, out_dim, bref, vref, bpg, vpg, bpi, vpi);
        free(x);
        free(xpad);
        free(w);
        free(wp);
        free(scales);
        free(bias);
        free(yref);
        free(ypg);
        free(ypi);
        return -1;
    }

    free(x);
    free(xpad);
    free(w);
    free(wp);
    free(scales);
    free(bias);
    free(yref);
    free(ypg);
    free(ypi);
    return 0;
}

static int check_bf16_matvec_case(int in_dim, int out_dim) {
    float *x = (float *)malloc((size_t)in_dim * sizeof(float));
    uint16_t *w = (uint16_t *)malloc((size_t)in_dim * (size_t)out_dim * sizeof(uint16_t));
    float *bias = (float *)malloc((size_t)out_dim * sizeof(float));
    float *yg = (float *)malloc((size_t)out_dim * sizeof(float));
    float *yi = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!x || !w || !bias || !yg || !yi) {
        free(x);
        free(w);
        free(bias);
        free(yg);
        free(yi);
        return -1;
    }

    for (int i = 0; i < in_dim; i++) x[i] = next_f32(0.002f);
    for (int i = 0; i < in_dim * out_dim; i++) w[i] = f32_to_bf16(next_f32(0.002f));
    for (int o = 0; o < out_dim; o++) bias[o] = next_f32(0.001f);

    nemo_bf16_matvec_fused_generic(yg, x, w, bias, in_dim, out_dim);
    nemo_bf16_matvec_fused_impl(yi, x, w, bias, in_dim, out_dim);
    for (int o = 0; o < out_dim; o++) {
        if (!nearly_equal(yg[o], yi[o], 5e-4f, 5e-4f)) {
            fprintf(stderr, "bf16 matvec mismatch in=%d out=%d o=%d generic=%g impl=%g\n",
                    in_dim, out_dim, o, yg[o], yi[o]);
            free(x);
            free(w);
            free(bias);
            free(yg);
            free(yi);
            return -1;
        }
    }

    int start = out_dim > 3 ? 1 : 0;
    int end = out_dim > 3 ? out_dim - 1 : out_dim;
    float vg = 0.0f, vi = 0.0f;
    int bg = nemo_argmax_bf16_range_generic(x, w, bias, in_dim, start, end, &vg);
    int bi = nemo_argmax_bf16_range_impl(x, w, bias, in_dim, start, end, &vi);
    if (bg != bi || !nearly_equal(vg, vi, 5e-4f, 5e-4f)) {
        fprintf(stderr, "bf16 argmax mismatch in=%d out=%d generic=(%d,%g) impl=(%d,%g)\n",
                in_dim, out_dim, bg, vg, bi, vi);
        free(x);
        free(w);
        free(bias);
        free(yg);
        free(yi);
        return -1;
    }

    free(x);
    free(w);
    free(bias);
    free(yg);
    free(yi);
    return 0;
}

int main(void) {
    static const int in_dims[] = {1, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129};
    static const int out_dims[] = {1, 3, 4, 5, 7, 16, 31};

    for (size_t i = 0; i < sizeof(in_dims) / sizeof(in_dims[0]); i++) {
        for (size_t o = 0; o < sizeof(out_dims) / sizeof(out_dims[0]); o++) {
            if (check_q8p_matvec_case(in_dims[i], out_dims[o]) != 0) return 1;
            if (check_bf16_matvec_case(in_dims[i], out_dims[o]) != 0) return 1;
        }
    }

    printf("kernel checks passed\n");
    return 0;
}
