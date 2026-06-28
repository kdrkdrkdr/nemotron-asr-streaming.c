/*
 * nemotron_asr_kernels_generic.c - portable scalar reference kernels.
 * f32 dot, packed Q8P int8 matvec/argmax (four-output-row tiles), attention
 * score, AXPY, radix-2 FFT power spectrum, and pointwise preconv. Serves as the
 * fallback backend and the numerical reference for the SIMD variants.
 */
#include "nemotron_asr_kernels_impl.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline float dot_f32_generic_inline(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

float nemo_dot_f32_generic(const float *a, const float *b, int n) {
    return dot_f32_generic_inline(a, b, n);
}

static inline const int8_t *q8p_row_block_generic(const int8_t *w, int row,
                                                  int stride, int k) {
    const int tile = row >> 2;
    const int lane = row & 3;
    return w + (size_t)tile * 4u * (size_t)stride +
           (size_t)(k >> 4) * 64u + (size_t)lane * 16u;
}

static inline int32_t dot_i8p_generic_inline(const int8_t *x,
                                             const int8_t *w, int row,
                                             int stride) {
    int32_t sum = 0;
    for (int k = 0; k < stride; k += 16) {
        const int8_t *wr = q8p_row_block_generic(w, row, stride, k);
        for (int j = 0; j < 16; j++) {
            sum += (int32_t)x[k + j] * (int32_t)wr[j];
        }
    }
    return sum;
}

static inline void dot4_i8p_generic_inline(const int8_t *x, const int8_t *w,
                                           int row, int stride,
                                           int32_t *s0_out, int32_t *s1_out,
                                           int32_t *s2_out, int32_t *s3_out) {
    const int tile = row >> 2;
    const int8_t *tile_base = w + (size_t)tile * 4u * (size_t)stride;
    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int k = 0; k < stride; k += 16) {
        const int8_t *blk = tile_base + (size_t)(k >> 4) * 64u;
        const int8_t *w0 = blk;
        const int8_t *w1 = blk + 16;
        const int8_t *w2 = blk + 32;
        const int8_t *w3 = blk + 48;
        for (int j = 0; j < 16; j++) {
            int32_t xv = x[k + j];
            s0 += xv * (int32_t)w0[j];
            s1 += xv * (int32_t)w1[j];
            s2 += xv * (int32_t)w2[j];
            s3 += xv * (int32_t)w3[j];
        }
    }
    *s0_out = s0;
    *s1_out = s1;
    *s2_out = s2;
    *s3_out = s3;
}

void nemo_q8p_matvec_fused_generic(float *y, const int8_t *x_q8, float x_scale,
                                   const int8_t *w, const float *w_scales,
                                   const float *b, int stride, int start, int out_dim) {
    int i = 0;
    int o = start;
    for (; i < out_dim && (o & 3); i++, o++) {
        int32_t acc = dot_i8p_generic_inline(x_q8, w, o, stride);
        y[i] = (float)acc * x_scale * w_scales[o] + (b ? b[i] : 0.0f);
    }
    for (; i + 3 < out_dim; i += 4, o += 4) {
        int32_t s0, s1, s2, s3;
        dot4_i8p_generic_inline(x_q8, w, o, stride, &s0, &s1, &s2, &s3);
        y[i] = (float)s0 * x_scale * w_scales[o] + (b ? b[i] : 0.0f);
        y[i + 1] = (float)s1 * x_scale * w_scales[o + 1] + (b ? b[i + 1] : 0.0f);
        y[i + 2] = (float)s2 * x_scale * w_scales[o + 2] + (b ? b[i + 2] : 0.0f);
        y[i + 3] = (float)s3 * x_scale * w_scales[o + 3] + (b ? b[i + 3] : 0.0f);
    }
    for (; i < out_dim; i++, o++) {
        int32_t acc = dot_i8p_generic_inline(x_q8, w, o, stride);
        y[i] = (float)acc * x_scale * w_scales[o] + (b ? b[i] : 0.0f);
    }
}

int nemo_argmax_q8p_range_generic(const int8_t *x_q8, float x_scale,
                                  const int8_t *w, const float *w_scales,
                                  const float *b, int stride, int start, int end,
                                  float *best_val_out) {
    int best = start;
    float best_val = -3.4028234663852886e38f;
    int o = start;
    for (; o < end && (o & 3); o++) {
        int32_t acc = dot_i8p_generic_inline(x_q8, w, o, stride);
        float v = (float)acc * x_scale * w_scales[o] + (b ? b[o] : 0.0f);
        if (v > best_val) {
            best_val = v;
            best = o;
        }
    }
    for (; o + 3 < end; o += 4) {
        int32_t s0, s1, s2, s3;
        dot4_i8p_generic_inline(x_q8, w, o, stride, &s0, &s1, &s2, &s3);
        float v0 = (float)s0 * x_scale * w_scales[o] + (b ? b[o] : 0.0f);
        float v1 = (float)s1 * x_scale * w_scales[o + 1] + (b ? b[o + 1] : 0.0f);
        float v2 = (float)s2 * x_scale * w_scales[o + 2] + (b ? b[o + 2] : 0.0f);
        float v3 = (float)s3 * x_scale * w_scales[o + 3] + (b ? b[o + 3] : 0.0f);
        if (v0 > best_val) {
            best_val = v0;
            best = o;
        }
        if (v1 > best_val) {
            best_val = v1;
            best = o + 1;
        }
        if (v2 > best_val) {
            best_val = v2;
            best = o + 2;
        }
        if (v3 > best_val) {
            best_val = v3;
            best = o + 3;
        }
    }
    for (; o < end; o++) {
        int32_t acc = dot_i8p_generic_inline(x_q8, w, o, stride);
        float v = (float)acc * x_scale * w_scales[o] + (b ? b[o] : 0.0f);
        if (v > best_val) {
            best_val = v;
            best = o;
        }
    }
    if (best_val_out) *best_val_out = best_val;
    return best;
}

float nemo_attention_score_f32_generic(const float *q, const float *bias_u, const float *k,
                                       const float *bias_v, const float *p, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += (q[i] + bias_u[i]) * k[i] + (q[i] + bias_v[i]) * p[i];
    }
    return sum;
}

void nemo_matvec_f32_generic(float *y, const float *x, const float *w, const float *b,
                             int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        const float *wr = w + (size_t)o * in_dim;
        y[o] = dot_f32_generic_inline(x, wr, in_dim) + (b ? b[o] : 0.0f);
    }
}

int nemo_argmax_matvec_f32_generic(const float *x, const float *w, const float *b,
                                   int in_dim, int out_dim, float *best_val_out) {
    int best = 0;
    float best_val = -3.4028234663852886e38f;
    for (int o = 0; o < out_dim; o++) {
        const float *wr = w + (size_t)o * in_dim;
        float v = dot_f32_generic_inline(x, wr, in_dim) + (b ? b[o] : 0.0f);
        if (v > best_val) {
            best_val = v;
            best = o;
        }
    }
    if (best_val_out) *best_val_out = best_val;
    return best;
}

void nemo_vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n) {
    for (int i = 0; i < n; i++) dst[i] += alpha * src[i];
}

static unsigned bit_reverse9(unsigned x) {
    unsigned r = 0;
    for (int i = 0; i < 9; i++) {
        r = (r << 1) | (x & 1u);
        x >>= 1;
    }
    return r;
}

void nemo_fft512_power_f32_generic(float *power, const float *frame) {
    float re[512];
    float im[512];
    for (unsigned i = 0; i < 512; i++) {
        unsigned j = bit_reverse9(i);
        re[j] = frame[i];
        im[j] = 0.0f;
    }
    for (int len = 2; len <= 512; len <<= 1) {
        int half = len >> 1;
        double ang = -2.0 * M_PI / (double)len;
        float wlen_re = cosf((float)ang);
        float wlen_im = sinf((float)ang);
        for (int i = 0; i < 512; i += len) {
            float wr = 1.0f;
            float wi = 0.0f;
            for (int j = 0; j < half; j++) {
                int even = i + j;
                int odd = even + half;
                float ur = re[even];
                float ui = im[even];
                float vr = re[odd] * wr - im[odd] * wi;
                float vi = re[odd] * wi + im[odd] * wr;
                re[even] = ur + vr;
                im[even] = ui + vi;
                re[odd] = ur - vr;
                im[odd] = ui - vi;
                float nwr = wr * wlen_re - wi * wlen_im;
                wi = wr * wlen_im + wi * wlen_re;
                wr = nwr;
            }
        }
    }
    for (int k = 0; k <= 256; k++) power[k] = re[k] * re[k] + im[k] * im[k];
}

void nemo_preconv_emit_f32_generic(float *out, const float *history, const float *w, const float *b,
                                   int out_start, int out_t, int total_t,
                                   int c_in, int c_out, int f_in, int f_out,
                                   int k, int stride, int left, int groups) {
    int in_per_group = c_in / groups;
    int out_per_group = c_out / groups;
    for (int lot = 0; lot < out_t; lot++) {
        int ot = out_start + lot;
        for (int oc = 0; oc < c_out; oc++) {
            int g = oc / out_per_group;
            int ic0 = g * in_per_group;
            for (int of = 0; of < f_out; of++) {
                float sum = b ? b[oc] : 0.0f;
                for (int icg = 0; icg < in_per_group; icg++) {
                    int ic = ic0 + icg;
                    int w_ic = (groups == c_in && in_per_group == 1) ? 0 : icg;
                    for (int kt = 0; kt < k; kt++) {
                        int it = ot * stride + kt - left;
                        if (it < 0 || it >= total_t) continue;
                        for (int kf = 0; kf < k; kf++) {
                            int iff = of * stride + kf - left;
                            if (iff < 0 || iff >= f_in) continue;
                            size_t iidx = ((size_t)it * c_in + ic) * f_in + iff;
                            size_t widx = (((size_t)oc * in_per_group + w_ic) * k + kt) * k + kf;
                            sum += history[iidx] * w[widx];
                        }
                    }
                }
                out[((size_t)lot * c_out + oc) * f_out + of] = sum;
            }
        }
    }
}
