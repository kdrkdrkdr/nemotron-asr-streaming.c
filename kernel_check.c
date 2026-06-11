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

static int check_q8_matvec_case(int in_dim, int out_dim) {
    int8_t *x = (int8_t *)malloc((size_t)in_dim);
    int8_t *w = (int8_t *)malloc((size_t)in_dim * (size_t)out_dim);
    float *scales = (float *)malloc((size_t)out_dim * sizeof(float));
    float *bias = (float *)malloc((size_t)out_dim * sizeof(float));
    float *yg = (float *)malloc((size_t)out_dim * sizeof(float));
    float *yi = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!x || !w || !scales || !bias || !yg || !yi) {
        free(x);
        free(w);
        free(scales);
        free(bias);
        free(yg);
        free(yi);
        return -1;
    }

    for (int i = 0; i < in_dim; i++) x[i] = next_i8();
    for (int i = 0; i < in_dim * out_dim; i++) w[i] = next_i8();
    for (int o = 0; o < out_dim; o++) {
        scales[o] = 0.0003f + 0.00001f * (float)(o % 17);
        bias[o] = next_f32(0.001f);
    }
    float x_scale = 0.0007f;

    nemo_q8_matvec_fused_generic(yg, x, x_scale, w, scales, bias, in_dim, out_dim);
    nemo_q8_matvec_fused_impl(yi, x, x_scale, w, scales, bias, in_dim, out_dim);
    for (int o = 0; o < out_dim; o++) {
        if (!nearly_equal(yg[o], yi[o], 1e-5f, 1e-6f)) {
            fprintf(stderr, "q8 matvec mismatch in=%d out=%d o=%d generic=%g impl=%g\n",
                    in_dim, out_dim, o, yg[o], yi[o]);
            free(x);
            free(w);
            free(scales);
            free(bias);
            free(yg);
            free(yi);
            return -1;
        }
    }

    int start = out_dim > 3 ? 1 : 0;
    int end = out_dim > 3 ? out_dim - 1 : out_dim;
    float vg = 0.0f, vi = 0.0f;
    int bg = nemo_argmax_q8_range_generic(x, x_scale, w, scales, bias, in_dim, start, end, &vg);
    int bi = nemo_argmax_q8_range_impl(x, x_scale, w, scales, bias, in_dim, start, end, &vi);
    if (bg != bi || !nearly_equal(vg, vi, 1e-5f, 1e-6f)) {
        fprintf(stderr, "q8 argmax mismatch in=%d out=%d generic=(%d,%g) impl=(%d,%g)\n",
                in_dim, out_dim, bg, vg, bi, vi);
        free(x);
        free(w);
        free(scales);
        free(bias);
        free(yg);
        free(yi);
        return -1;
    }

    free(x);
    free(w);
    free(scales);
    free(bias);
    free(yg);
    free(yi);
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
            if (check_q8_matvec_case(in_dims[i], out_dims[o]) != 0) return 1;
            if (check_bf16_matvec_case(in_dims[i], out_dims[o]) != 0) return 1;
        }
    }

    printf("kernel checks passed\n");
    return 0;
}
