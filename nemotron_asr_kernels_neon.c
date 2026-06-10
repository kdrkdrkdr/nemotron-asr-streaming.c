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
