/*
 * nemotron_asr_kernels_neon.c - ARM NEON hot kernels.
 * Baseline-NEON f32 dot and packed Q8P int8 matvec/argmax (vmull_s8 +
 * vpadalq_s16, no dotprod/i8mm), attention score, and AXPY. Compiled only under
 * __ARM_NEON; FFT and preconv fall back to the generic backend.
 */
#include "nemotron_asr_kernels_impl.h"

#ifdef __ARM_NEON

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __aarch64__
#define NEMO_FMAQ_F32(acc, a, b) vfmaq_f32((acc), (a), (b))
#else
#define NEMO_FMAQ_F32(acc, a, b) vmlaq_f32((acc), (a), (b))
#endif

static inline float hsum_f32x4(float32x4_t v) {
#ifdef __aarch64__
    return vaddvq_f32(v);
#else
    float tmp[4];
    vst1q_f32(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
}

static inline int32_t hsum_s32x4(int32x4_t v) {
#ifdef __aarch64__
    return vaddvq_s32(v);
#else
    int32_t tmp[4];
    vst1q_s32(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
}

static inline float dot_f32_neon_inline(const float *a, const float *b, int n) {
    int i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    for (; i + 16 <= n; i += 16) {
        acc0 = NEMO_FMAQ_F32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = NEMO_FMAQ_F32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc2 = NEMO_FMAQ_F32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        acc3 = NEMO_FMAQ_F32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    for (; i + 4 <= n; i += 4) {
        acc0 = NEMO_FMAQ_F32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float sum = hsum_f32x4(acc0);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

float nemo_dot_f32_neon(const float *a, const float *b, int n) {
    return dot_f32_neon_inline(a, b, n);
}

/* Q8P packed layout: see nemo_q8p_row_block in nemotron_asr_kernels.c. */
static inline int32_t dot_i8p_neon_inline(const int8_t *x, const int8_t *w,
                                          int row, int stride) {
    const int tile = row >> 2;
    const int lane = row & 3;
    const int8_t *wr = w + (size_t)tile * 4u * (size_t)stride + (size_t)lane * 16u;
    int32x4_t acc = vdupq_n_s32(0);
    for (int k = 0; k < stride; k += 16, wr += 64) {
        int8x16_t xv = vld1q_s8(x + k);
        int8x16_t wv = vld1q_s8(wr);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(xv), vget_low_s8(wv)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(xv), vget_high_s8(wv)));
    }
    return hsum_s32x4(acc);
}

static inline void dot4_i8p_neon_inline(const int8_t *x, const int8_t *w,
                                        int row, int stride,
                                        int32_t *s0_out, int32_t *s1_out,
                                        int32_t *s2_out, int32_t *s3_out) {
    const int tile = row >> 2;
    const int8_t *blk = w + (size_t)tile * 4u * (size_t)stride;
    int32x4_t a0 = vdupq_n_s32(0);
    int32x4_t a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0);
    int32x4_t a3 = vdupq_n_s32(0);
    for (int k = 0; k < stride; k += 16, blk += 64) {
        int8x16_t xv = vld1q_s8(x + k);
        int8x8_t xlo = vget_low_s8(xv);
        int8x8_t xhi = vget_high_s8(xv);
        int8x16_t wv0 = vld1q_s8(blk);
        int8x16_t wv1 = vld1q_s8(blk + 16);
        int8x16_t wv2 = vld1q_s8(blk + 32);
        int8x16_t wv3 = vld1q_s8(blk + 48);
        a0 = vpadalq_s16(a0, vmull_s8(xlo, vget_low_s8(wv0)));
        a0 = vpadalq_s16(a0, vmull_s8(xhi, vget_high_s8(wv0)));
        a1 = vpadalq_s16(a1, vmull_s8(xlo, vget_low_s8(wv1)));
        a1 = vpadalq_s16(a1, vmull_s8(xhi, vget_high_s8(wv1)));
        a2 = vpadalq_s16(a2, vmull_s8(xlo, vget_low_s8(wv2)));
        a2 = vpadalq_s16(a2, vmull_s8(xhi, vget_high_s8(wv2)));
        a3 = vpadalq_s16(a3, vmull_s8(xlo, vget_low_s8(wv3)));
        a3 = vpadalq_s16(a3, vmull_s8(xhi, vget_high_s8(wv3)));
    }
    *s0_out = hsum_s32x4(a0);
    *s1_out = hsum_s32x4(a1);
    *s2_out = hsum_s32x4(a2);
    *s3_out = hsum_s32x4(a3);
}

void nemo_q8p_matvec_fused_neon(float *y, const int8_t *x_q8, float x_scale,
                                const int8_t *w, const float *w_scales,
                                const float *b, int stride, int start, int out_dim) {
    int i = 0;
    int o = start;
    for (; i < out_dim && (o & 3); i++, o++) {
        int32_t acc = dot_i8p_neon_inline(x_q8, w, o, stride);
        y[i] = (float)acc * x_scale * w_scales[o] + (b ? b[i] : 0.0f);
    }
    for (; i + 3 < out_dim; i += 4, o += 4) {
        int32_t s0, s1, s2, s3;
        dot4_i8p_neon_inline(x_q8, w, o, stride, &s0, &s1, &s2, &s3);
        y[i] = (float)s0 * x_scale * w_scales[o] + (b ? b[i] : 0.0f);
        y[i + 1] = (float)s1 * x_scale * w_scales[o + 1] + (b ? b[i + 1] : 0.0f);
        y[i + 2] = (float)s2 * x_scale * w_scales[o + 2] + (b ? b[i + 2] : 0.0f);
        y[i + 3] = (float)s3 * x_scale * w_scales[o + 3] + (b ? b[i + 3] : 0.0f);
    }
    for (; i < out_dim; i++, o++) {
        int32_t acc = dot_i8p_neon_inline(x_q8, w, o, stride);
        y[i] = (float)acc * x_scale * w_scales[o] + (b ? b[i] : 0.0f);
    }
}

int nemo_argmax_q8p_range_neon(const int8_t *x_q8, float x_scale,
                               const int8_t *w, const float *w_scales,
                               const float *b, int stride, int start, int end,
                               float *best_val_out) {
    int best = start;
    float best_val = -3.4028234663852886e38f;
    int o = start;
    for (; o < end && (o & 3); o++) {
        int32_t acc = dot_i8p_neon_inline(x_q8, w, o, stride);
        float v = (float)acc * x_scale * w_scales[o] + (b ? b[o] : 0.0f);
        if (v > best_val) {
            best_val = v;
            best = o;
        }
    }
    for (; o + 3 < end; o += 4) {
        int32_t s0, s1, s2, s3;
        dot4_i8p_neon_inline(x_q8, w, o, stride, &s0, &s1, &s2, &s3);
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
        int32_t acc = dot_i8p_neon_inline(x_q8, w, o, stride);
        float v = (float)acc * x_scale * w_scales[o] + (b ? b[o] : 0.0f);
        if (v > best_val) {
            best_val = v;
            best = o;
        }
    }
    if (best_val_out) *best_val_out = best_val;
    return best;
}

float nemo_attention_score_f32_neon(const float *q, const float *bias_u, const float *k,
                                    const float *bias_v, const float *p, int n) {
    int i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    for (; i + 16 <= n; i += 16) {
        float32x4_t q0 = vld1q_f32(q + i);
        float32x4_t q1 = vld1q_f32(q + i + 4);
        float32x4_t q2 = vld1q_f32(q + i + 8);
        float32x4_t q3 = vld1q_f32(q + i + 12);
        acc0 = NEMO_FMAQ_F32(acc0, vaddq_f32(q0, vld1q_f32(bias_u + i)), vld1q_f32(k + i));
        acc1 = NEMO_FMAQ_F32(acc1, vaddq_f32(q1, vld1q_f32(bias_u + i + 4)), vld1q_f32(k + i + 4));
        acc2 = NEMO_FMAQ_F32(acc2, vaddq_f32(q2, vld1q_f32(bias_u + i + 8)), vld1q_f32(k + i + 8));
        acc3 = NEMO_FMAQ_F32(acc3, vaddq_f32(q3, vld1q_f32(bias_u + i + 12)), vld1q_f32(k + i + 12));
        acc0 = NEMO_FMAQ_F32(acc0, vaddq_f32(q0, vld1q_f32(bias_v + i)), vld1q_f32(p + i));
        acc1 = NEMO_FMAQ_F32(acc1, vaddq_f32(q1, vld1q_f32(bias_v + i + 4)), vld1q_f32(p + i + 4));
        acc2 = NEMO_FMAQ_F32(acc2, vaddq_f32(q2, vld1q_f32(bias_v + i + 8)), vld1q_f32(p + i + 8));
        acc3 = NEMO_FMAQ_F32(acc3, vaddq_f32(q3, vld1q_f32(bias_v + i + 12)), vld1q_f32(p + i + 12));
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    for (; i + 4 <= n; i += 4) {
        float32x4_t q0 = vld1q_f32(q + i);
        acc0 = NEMO_FMAQ_F32(acc0, vaddq_f32(q0, vld1q_f32(bias_u + i)), vld1q_f32(k + i));
        acc0 = NEMO_FMAQ_F32(acc0, vaddq_f32(q0, vld1q_f32(bias_v + i)), vld1q_f32(p + i));
    }
    float sum = hsum_f32x4(acc0);
    for (; i < n; i++) {
        sum += (q[i] + bias_u[i]) * k[i] + (q[i] + bias_v[i]) * p[i];
    }
    return sum;
}

static void nemo_matvec_f32_neon(float *y, const float *x, const float *w, const float *b,
                                 int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        y[o] = dot_f32_neon_inline(x, w + (size_t)o * in_dim, in_dim) + (b ? b[o] : 0.0f);
    }
}

void nemo_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n) {
    int i = 0;
    float32x4_t a = vdupq_n_f32(alpha);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(dst + i, NEMO_FMAQ_F32(vld1q_f32(dst + i), vld1q_f32(src + i), a));
        vst1q_f32(dst + i + 4, NEMO_FMAQ_F32(vld1q_f32(dst + i + 4), vld1q_f32(src + i + 4), a));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
}

void nemo_preconv_emit_f32_neon(float *out, const float *history, const float *w, const float *b,
                                int out_start, int out_t, int total_t,
                                int c_in, int c_out, int f_in, int f_out,
                                int k, int stride, int left, int groups) {
    if (k == 1 && stride == 1 && left == 0 && groups == 1) {
        float *xv = (float *)malloc((size_t)c_in * sizeof(float));
        float *yv = (float *)malloc((size_t)c_out * sizeof(float));
        if (xv && yv) {
            for (int lot = 0; lot < out_t; lot++) {
                int ot = out_start + lot;
                if (ot < 0 || ot >= total_t) continue;
                for (int of = 0; of < f_out; of++) {
                    for (int ic = 0; ic < c_in; ic++) {
                        xv[ic] = history[((size_t)ot * c_in + ic) * f_in + of];
                    }
                    nemo_matvec_f32_neon(yv, xv, w, b, c_in, c_out);
                    for (int oc = 0; oc < c_out; oc++) {
                        out[((size_t)lot * c_out + oc) * f_out + of] = yv[oc];
                    }
                }
            }
            free(xv);
            free(yv);
            return;
        }
        free(xv);
        free(yv);
    }
    nemo_preconv_emit_f32_generic(out, history, w, b, out_start, out_t, total_t,
                                  c_in, c_out, f_in, f_out, k, stride, left, groups);
}

void nemo_fft512_power_f32_neon(float *power, const float *frame) {
    nemo_fft512_power_f32_generic(power, frame);
}

#endif
