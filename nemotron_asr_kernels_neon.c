#include "nemotron_asr_kernels_impl.h"

#ifdef __ARM_NEON

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

float nemo_dot_bf16_f32_neon(const float *a, const uint16_t *b, int n) {
    int i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    for (; i + 16 <= n; i += 16) {
        uint16x8_t b0 = vld1q_u16(b + i);
        uint16x8_t b1 = vld1q_u16(b + i + 8);
        acc0 = NEMO_FMAQ_F32(acc0,
                              vld1q_f32(a + i),
                              vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b0), 16)));
        acc1 = NEMO_FMAQ_F32(acc1,
                              vld1q_f32(a + i + 4),
                              vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b0), 16)));
        acc2 = NEMO_FMAQ_F32(acc2,
                              vld1q_f32(a + i + 8),
                              vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b1), 16)));
        acc3 = NEMO_FMAQ_F32(acc3,
                              vld1q_f32(a + i + 12),
                              vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b1), 16)));
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        uint16x8_t bv = vld1q_u16(b + i);
        acc0 = NEMO_FMAQ_F32(acc0,
                              vld1q_f32(a + i),
                              vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bv), 16)));
        acc1 = NEMO_FMAQ_F32(acc1,
                              vld1q_f32(a + i + 4),
                              vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bv), 16)));
    }
    float sum = hsum_f32x4(vaddq_f32(acc0, acc1));
    for (; i < n; i++) {
        uint32_t bits = (uint32_t)b[i] << 16;
        float wv;
        memcpy(&wv, &bits, sizeof(wv));
        sum += a[i] * wv;
    }
    return sum;
}

void nemo_bf16_matvec_fused_neon(float *y, const float *x, const uint16_t *w,
                                 const float *b, int in_dim, int out_dim) {
    int o = 0;
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = w + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        float s0 = b ? b[o] : 0.0f;
        float s1 = b ? b[o + 1] : 0.0f;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
        float32x4_t b2 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            float32x4_t x2 = vld1q_f32(x + k + 8);
            float32x4_t x3 = vld1q_f32(x + k + 12);
            uint16x8_t r0a = vld1q_u16(w0 + k);
            uint16x8_t r0b = vld1q_u16(w0 + k + 8);
            uint16x8_t r1a = vld1q_u16(w1 + k);
            uint16x8_t r1b = vld1q_u16(w1 + k + 8);
            a0 = NEMO_FMAQ_F32(a0, x0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)));
            a1 = NEMO_FMAQ_F32(a1, x1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)));
            a2 = NEMO_FMAQ_F32(a2, x2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)));
            a3 = NEMO_FMAQ_F32(a3, x3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)));
            b0 = NEMO_FMAQ_F32(b0, x0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1a), 16)));
            b1 = NEMO_FMAQ_F32(b1, x1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1a), 16)));
            b2 = NEMO_FMAQ_F32(b2, x2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1b), 16)));
            b3 = NEMO_FMAQ_F32(b3, x3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1b), 16)));
        }
        for (; k + 8 <= in_dim; k += 8) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            uint16x8_t r0 = vld1q_u16(w0 + k);
            uint16x8_t r1 = vld1q_u16(w1 + k);
            a0 = NEMO_FMAQ_F32(a0, x0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0), 16)));
            a1 = NEMO_FMAQ_F32(a1, x1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0), 16)));
            b0 = NEMO_FMAQ_F32(b0, x0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1), 16)));
            b1 = NEMO_FMAQ_F32(b1, x1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1), 16)));
        }
        s0 += hsum_f32x4(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        s1 += hsum_f32x4(vaddq_f32(vaddq_f32(b0, b1), vaddq_f32(b2, b3)));
        for (; k < in_dim; k++) {
            uint32_t bits0 = (uint32_t)w0[k] << 16;
            uint32_t bits1 = (uint32_t)w1[k] << 16;
            float wv0, wv1;
            memcpy(&wv0, &bits0, sizeof(wv0));
            memcpy(&wv1, &bits1, sizeof(wv1));
            s0 += x[k] * wv0;
            s1 += x[k] * wv1;
        }
        y[o] = s0;
        y[o + 1] = s1;
    }
    for (; o < out_dim; o++) {
        y[o] = nemo_dot_bf16_f32_neon(x, w + (size_t)o * in_dim, in_dim) + (b ? b[o] : 0.0f);
    }
}

int nemo_argmax_bf16_range_neon(const float *x, const uint16_t *w, const float *b,
                                int in_dim, int start, int end, float *best_val_out) {
    int best = start;
    float best_val = -3.4028234663852886e38f;
    for (int o = start; o < end; o++) {
        float v = nemo_dot_bf16_f32_neon(x, w + (size_t)o * in_dim, in_dim) +
                  (b ? b[o] : 0.0f);
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

void nemo_matvec_f32_neon(float *y, const float *x, const float *w, const float *b,
                          int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        y[o] = dot_f32_neon_inline(x, w + (size_t)o * in_dim, in_dim) + (b ? b[o] : 0.0f);
    }
}

int nemo_argmax_matvec_f32_neon(const float *x, const float *w, const float *b,
                                int in_dim, int out_dim, float *best_val_out) {
    int best = 0;
    float best_val = -3.4028234663852886e38f;
    for (int o = 0; o < out_dim; o++) {
        float v = dot_f32_neon_inline(x, w + (size_t)o * in_dim, in_dim) + (b ? b[o] : 0.0f);
        if (v > best_val) {
            best_val = v;
            best = o;
        }
    }
    if (best_val_out) *best_val_out = best_val;
    return best;
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
