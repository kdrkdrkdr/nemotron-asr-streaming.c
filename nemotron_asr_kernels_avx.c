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

float nemo_dot_bf16_f32_avx(const float *a, const uint16_t *b, int n) {
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i b1 = _mm256_loadu_si256((const __m256i *)(b + i + 16));
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),
                                _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(b0), 16)),
                                acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16),
                                _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(b1), 16)),
                                acc1);
    }
    for (; i + 16 <= n; i += 16) {
        __m256i bv = _mm256_loadu_si256((const __m256i *)(b + i));
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),
                                _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bv), 16)),
                                acc0);
    }
    float sum = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
#else
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        __m128i b0 = _mm_loadu_si128((const __m128i *)(b + i));
        __m128i b1 = _mm_loadu_si128((const __m128i *)(b + i + 8));
        __m128i b2 = _mm_loadu_si128((const __m128i *)(b + i + 16));
        __m128i b3 = _mm_loadu_si128((const __m128i *)(b + i + 24));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                                _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b0), 16)),
                                acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                                _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b1), 16)),
                                acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
                                _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b2), 16)),
                                acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
                                _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b3), 16)),
                                acc3);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    for (; i + 8 <= n; i += 8) {
        __m128i bv = _mm_loadu_si128((const __m128i *)(b + i));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                                _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bv), 16)),
                                acc0);
    }
    float sum = hsum_m256(acc0);
#endif
    for (; i < n; i++) {
        uint32_t bits = (uint32_t)b[i] << 16;
        float wv;
        memcpy(&wv, &bits, sizeof(wv));
        sum += a[i] * wv;
    }
    return sum;
}

void nemo_bf16_matvec_fused_avx(float *y, const float *x, const uint16_t *w,
                                const float *b, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        y[o] = nemo_dot_bf16_f32_avx(x, w + (size_t)o * in_dim, in_dim) +
               (b ? b[o] : 0.0f);
    }
}

int nemo_argmax_bf16_range_avx(const float *x, const uint16_t *w, const float *b,
                               int in_dim, int start, int end, float *best_val_out) {
    int best = start;
    float best_val = -3.4028234663852886e38f;
    for (int o = start; o < end; o++) {
        float v = nemo_dot_bf16_f32_avx(x, w + (size_t)o * in_dim, in_dim) +
                  (b ? b[o] : 0.0f);
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
