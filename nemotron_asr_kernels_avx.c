#include "nemotron_asr_kernels_impl.h"

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline float hsum_m256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static inline int32_t hsum_m256i_epi32(__m256i v) __attribute__((unused));
static inline int32_t hsum_m256i_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(sum);
}

static inline float dot_f32_avx_inline(const float *a, const float *b, int n) {
#if defined(__AVX512F__)
    int i = 0;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), acc3);
    }
    __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#else
    int i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    float sum = hsum_m256(acc0);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

float nemo_dot_f32_avx(const float *a, const float *b, int n) {
    return dot_f32_avx_inline(a, b, n);
}

static inline int32_t dot_i8p_avx_inline(const int8_t *x, const int8_t *w,
                                         int row, int stride) {
    const int tile = row >> 2;
    const int lane = row & 3;
    const int8_t *wr = w + (size_t)tile * 4u * (size_t)stride + (size_t)lane * 16u;
    __m256i acc = _mm256_setzero_si256();
    for (int k = 0; k < stride; k += 16, wr += 64) {
        __m256i xv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x + k)));
        __m256i wv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)wr));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(xv, wv));
    }
    return hsum_m256i_epi32(acc);
}

static inline void dot4_i8p_avx_inline(const int8_t *x, const int8_t *w,
                                       int row, int stride,
                                       int32_t *s0_out, int32_t *s1_out,
                                       int32_t *s2_out, int32_t *s3_out) {
    const int tile = row >> 2;
    const int8_t *blk = w + (size_t)tile * 4u * (size_t)stride;
    __m256i a0 = _mm256_setzero_si256();
    __m256i a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256();
    __m256i a3 = _mm256_setzero_si256();
    for (int k = 0; k < stride; k += 16, blk += 64) {
        __m256i xv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x + k)));
        a0 = _mm256_add_epi32(a0,
                              _mm256_madd_epi16(xv, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)blk))));
        a1 = _mm256_add_epi32(a1,
                              _mm256_madd_epi16(xv, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(blk + 16)))));
        a2 = _mm256_add_epi32(a2,
                              _mm256_madd_epi16(xv, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(blk + 32)))));
        a3 = _mm256_add_epi32(a3,
                              _mm256_madd_epi16(xv, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(blk + 48)))));
    }
    *s0_out = hsum_m256i_epi32(a0);
    *s1_out = hsum_m256i_epi32(a1);
    *s2_out = hsum_m256i_epi32(a2);
    *s3_out = hsum_m256i_epi32(a3);
}

void nemo_q8p_matvec_fused_avx(float *y, const int8_t *x_q8, float x_scale,
                               const int8_t *w, const float *w_scales,
                               const float *b, int stride, int start, int out_dim) {
    int i = 0;
    int o = start;
    for (; i < out_dim && (o & 3); i++, o++) {
        int32_t acc = dot_i8p_avx_inline(x_q8, w, o, stride);
        y[i] = (float)acc * x_scale * w_scales[o] + (b ? b[i] : 0.0f);
    }
    for (; i + 3 < out_dim; i += 4, o += 4) {
        int32_t s0, s1, s2, s3;
        dot4_i8p_avx_inline(x_q8, w, o, stride, &s0, &s1, &s2, &s3);
        y[i] = (float)s0 * x_scale * w_scales[o] + (b ? b[i] : 0.0f);
        y[i + 1] = (float)s1 * x_scale * w_scales[o + 1] + (b ? b[i + 1] : 0.0f);
        y[i + 2] = (float)s2 * x_scale * w_scales[o + 2] + (b ? b[i + 2] : 0.0f);
        y[i + 3] = (float)s3 * x_scale * w_scales[o + 3] + (b ? b[i + 3] : 0.0f);
    }
    for (; i < out_dim; i++, o++) {
        int32_t acc = dot_i8p_avx_inline(x_q8, w, o, stride);
        y[i] = (float)acc * x_scale * w_scales[o] + (b ? b[i] : 0.0f);
    }
}

int nemo_argmax_q8p_range_avx(const int8_t *x_q8, float x_scale,
                              const int8_t *w, const float *w_scales,
                              const float *b, int stride, int start, int end,
                              float *best_val_out) {
    int best = start;
    float best_val = -3.4028234663852886e38f;
    int o = start;
    for (; o < end && (o & 3); o++) {
        int32_t acc = dot_i8p_avx_inline(x_q8, w, o, stride);
        float v = (float)acc * x_scale * w_scales[o] + (b ? b[o] : 0.0f);
        if (v > best_val) {
            best_val = v;
            best = o;
        }
    }
    for (; o + 3 < end; o += 4) {
        int32_t s0, s1, s2, s3;
        dot4_i8p_avx_inline(x_q8, w, o, stride, &s0, &s1, &s2, &s3);
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
        int32_t acc = dot_i8p_avx_inline(x_q8, w, o, stride);
        float v = (float)acc * x_scale * w_scales[o] + (b ? b[o] : 0.0f);
        if (v > best_val) {
            best_val = v;
            best = o;
        }
    }
    if (best_val_out) *best_val_out = best_val;
    return best;
}

float nemo_attention_score_f32_avx(const float *q, const float *bias_u, const float *k,
                                   const float *bias_v, const float *p, int n) {
    int i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m256 q0 = _mm256_loadu_ps(q + i);
        __m256 q1 = _mm256_loadu_ps(q + i + 8);
        acc0 = _mm256_fmadd_ps(_mm256_add_ps(q0, _mm256_loadu_ps(bias_u + i)),
                               _mm256_loadu_ps(k + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_add_ps(q1, _mm256_loadu_ps(bias_u + i + 8)),
                               _mm256_loadu_ps(k + i + 8), acc1);
        acc0 = _mm256_fmadd_ps(_mm256_add_ps(q0, _mm256_loadu_ps(bias_v + i)),
                               _mm256_loadu_ps(p + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_add_ps(q1, _mm256_loadu_ps(bias_v + i + 8)),
                               _mm256_loadu_ps(p + i + 8), acc1);
    }
    float sum = hsum_m256(_mm256_add_ps(acc0, acc1));
    for (; i < n; i++) {
        sum += (q[i] + bias_u[i]) * k[i] + (q[i] + bias_v[i]) * p[i];
    }
    return sum;
}

void nemo_matvec_f32_avx(float *y, const float *x, const float *w, const float *b,
                         int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        y[o] = dot_f32_avx_inline(x, w + (size_t)o * in_dim, in_dim) + (b ? b[o] : 0.0f);
    }
}

int nemo_argmax_matvec_f32_avx(const float *x, const float *w, const float *b,
                               int in_dim, int out_dim, float *best_val_out) {
    int best = 0;
    float best_val = -3.4028234663852886e38f;
    for (int o = 0; o < out_dim; o++) {
        float v = dot_f32_avx_inline(x, w + (size_t)o * in_dim, in_dim) + (b ? b[o] : 0.0f);
        if (v > best_val) {
            best_val = v;
            best = o;
        }
    }
    if (best_val_out) *best_val_out = best_val;
    return best;
}

void nemo_vec_axpy_inplace_avx(float *dst, const float *src, float alpha, int n) {
    int i = 0;
    __m256 a = _mm256_set1_ps(alpha);
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(dst + i,
                         _mm256_fmadd_ps(_mm256_loadu_ps(src + i), a, _mm256_loadu_ps(dst + i)));
        _mm256_storeu_ps(dst + i + 8,
                         _mm256_fmadd_ps(_mm256_loadu_ps(src + i + 8), a, _mm256_loadu_ps(dst + i + 8)));
        _mm256_storeu_ps(dst + i + 16,
                         _mm256_fmadd_ps(_mm256_loadu_ps(src + i + 16), a, _mm256_loadu_ps(dst + i + 16)));
        _mm256_storeu_ps(dst + i + 24,
                         _mm256_fmadd_ps(_mm256_loadu_ps(src + i + 24), a, _mm256_loadu_ps(dst + i + 24)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i,
                         _mm256_fmadd_ps(_mm256_loadu_ps(src + i), a, _mm256_loadu_ps(dst + i)));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
}

void nemo_preconv_emit_f32_avx(float *out, const float *history, const float *w, const float *b,
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
                    nemo_matvec_f32_avx(yv, xv, w, b, c_in, c_out);
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

void nemo_fft512_power_f32_avx(float *power, const float *frame) {
    nemo_fft512_power_f32_generic(power, frame);
}

#endif
