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

static inline float bf16_to_f32_scalar(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
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
        sum += a[i] * bf16_to_f32_scalar(b[i]);
    }
    return sum;
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
static inline __m512 bf16x16_to_f32_avx512(const uint16_t *src) {
    __m256i raw = _mm256_loadu_si256((const __m256i *)src);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
}

static inline void bf16_dot4_avx512(const float *x,
                                    const uint16_t *w0, const uint16_t *w1,
                                    const uint16_t *w2, const uint16_t *w3,
                                    int in_dim, float *s0_out, float *s1_out,
                                    float *s2_out, float *s3_out) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    __m512 a4 = _mm512_setzero_ps(), a5 = _mm512_setzero_ps();
    __m512 a6 = _mm512_setzero_ps(), a7 = _mm512_setzero_ps();
    int k = 0;
    for (; k + 32 <= in_dim; k += 32) {
        __m512 x0 = _mm512_loadu_ps(x + k);
        __m512 x1 = _mm512_loadu_ps(x + k + 16);
        a0 = _mm512_fmadd_ps(x0, bf16x16_to_f32_avx512(w0 + k), a0);
        a1 = _mm512_fmadd_ps(x1, bf16x16_to_f32_avx512(w0 + k + 16), a1);
        a2 = _mm512_fmadd_ps(x0, bf16x16_to_f32_avx512(w1 + k), a2);
        a3 = _mm512_fmadd_ps(x1, bf16x16_to_f32_avx512(w1 + k + 16), a3);
        a4 = _mm512_fmadd_ps(x0, bf16x16_to_f32_avx512(w2 + k), a4);
        a5 = _mm512_fmadd_ps(x1, bf16x16_to_f32_avx512(w2 + k + 16), a5);
        a6 = _mm512_fmadd_ps(x0, bf16x16_to_f32_avx512(w3 + k), a6);
        a7 = _mm512_fmadd_ps(x1, bf16x16_to_f32_avx512(w3 + k + 16), a7);
    }
    for (; k + 16 <= in_dim; k += 16) {
        __m512 xv = _mm512_loadu_ps(x + k);
        a0 = _mm512_fmadd_ps(xv, bf16x16_to_f32_avx512(w0 + k), a0);
        a2 = _mm512_fmadd_ps(xv, bf16x16_to_f32_avx512(w1 + k), a2);
        a4 = _mm512_fmadd_ps(xv, bf16x16_to_f32_avx512(w2 + k), a4);
        a6 = _mm512_fmadd_ps(xv, bf16x16_to_f32_avx512(w3 + k), a6);
    }
    float s0 = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    float s1 = _mm512_reduce_add_ps(_mm512_add_ps(a2, a3));
    float s2 = _mm512_reduce_add_ps(_mm512_add_ps(a4, a5));
    float s3 = _mm512_reduce_add_ps(_mm512_add_ps(a6, a7));
    for (; k < in_dim; k++) {
        float xk = x[k];
        s0 += xk * bf16_to_f32_scalar(w0[k]);
        s1 += xk * bf16_to_f32_scalar(w1[k]);
        s2 += xk * bf16_to_f32_scalar(w2[k]);
        s3 += xk * bf16_to_f32_scalar(w3[k]);
    }
    *s0_out = s0;
    *s1_out = s1;
    *s2_out = s2;
    *s3_out = s3;
}

static inline void bf16_dot2_avx512(const float *x, const uint16_t *w0,
                                    const uint16_t *w1, int in_dim,
                                    float *s0_out, float *s1_out) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    int k = 0;
    for (; k + 32 <= in_dim; k += 32) {
        __m512 x0 = _mm512_loadu_ps(x + k);
        __m512 x1 = _mm512_loadu_ps(x + k + 16);
        a0 = _mm512_fmadd_ps(x0, bf16x16_to_f32_avx512(w0 + k), a0);
        a1 = _mm512_fmadd_ps(x1, bf16x16_to_f32_avx512(w0 + k + 16), a1);
        a2 = _mm512_fmadd_ps(x0, bf16x16_to_f32_avx512(w1 + k), a2);
        a3 = _mm512_fmadd_ps(x1, bf16x16_to_f32_avx512(w1 + k + 16), a3);
    }
    for (; k + 16 <= in_dim; k += 16) {
        __m512 xv = _mm512_loadu_ps(x + k);
        a0 = _mm512_fmadd_ps(xv, bf16x16_to_f32_avx512(w0 + k), a0);
        a2 = _mm512_fmadd_ps(xv, bf16x16_to_f32_avx512(w1 + k), a2);
    }
    float s0 = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    float s1 = _mm512_reduce_add_ps(_mm512_add_ps(a2, a3));
    for (; k < in_dim; k++) {
        float xk = x[k];
        s0 += xk * bf16_to_f32_scalar(w0[k]);
        s1 += xk * bf16_to_f32_scalar(w1[k]);
    }
    *s0_out = s0;
    *s1_out = s1;
}
#else
static inline __m256 bf16x8_to_f32_avx2(__m128i raw) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

static inline void bf16x16_to_f32_avx2(const uint16_t *src, __m256 *lo, __m256 *hi) {
    __m256i raw = _mm256_loadu_si256((const __m256i *)src);
    *lo = bf16x8_to_f32_avx2(_mm256_castsi256_si128(raw));
    *hi = bf16x8_to_f32_avx2(_mm256_extracti128_si256(raw, 1));
}

static inline void bf16_dot2_avx2(const float *x, const uint16_t *w0,
                                  const uint16_t *w1, int in_dim,
                                  float *s0_out, float *s1_out) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 16 <= in_dim; k += 16) {
        __m256 x0 = _mm256_loadu_ps(x + k);
        __m256 x1 = _mm256_loadu_ps(x + k + 8);
        __m256 wlo, whi;
        bf16x16_to_f32_avx2(w0 + k, &wlo, &whi);
        a0 = _mm256_fmadd_ps(x0, wlo, a0);
        a1 = _mm256_fmadd_ps(x1, whi, a1);
        bf16x16_to_f32_avx2(w1 + k, &wlo, &whi);
        a2 = _mm256_fmadd_ps(x0, wlo, a2);
        a3 = _mm256_fmadd_ps(x1, whi, a3);
    }
    float s0 = hsum_m256(_mm256_add_ps(a0, a1));
    float s1 = hsum_m256(_mm256_add_ps(a2, a3));
    for (; k < in_dim; k++) {
        float xk = x[k];
        s0 += xk * bf16_to_f32_scalar(w0[k]);
        s1 += xk * bf16_to_f32_scalar(w1[k]);
    }
    *s0_out = s0;
    *s1_out = s1;
}
#endif

void nemo_bf16_matvec_fused_avx(float *y, const float *x, const uint16_t *w,
                                const float *b, int in_dim, int out_dim) {
    int o = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (; o + 3 < out_dim; o += 4) {
        const uint16_t *w0 = w + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        const uint16_t *w2 = w1 + in_dim;
        const uint16_t *w3 = w2 + in_dim;
        float s0, s1, s2, s3;
        bf16_dot4_avx512(x, w0, w1, w2, w3, in_dim, &s0, &s1, &s2, &s3);
        y[o] = s0 + (b ? b[o] : 0.0f);
        y[o + 1] = s1 + (b ? b[o + 1] : 0.0f);
        y[o + 2] = s2 + (b ? b[o + 2] : 0.0f);
        y[o + 3] = s3 + (b ? b[o + 3] : 0.0f);
    }
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = w + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        float s0, s1;
        bf16_dot2_avx512(x, w0, w1, in_dim, &s0, &s1);
        y[o] = s0 + (b ? b[o] : 0.0f);
        y[o + 1] = s1 + (b ? b[o + 1] : 0.0f);
    }
#else
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = w + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        float s0, s1;
        bf16_dot2_avx2(x, w0, w1, in_dim, &s0, &s1);
        y[o] = s0 + (b ? b[o] : 0.0f);
        y[o + 1] = s1 + (b ? b[o + 1] : 0.0f);
    }
#endif
    for (; o < out_dim; o++) {
        y[o] = nemo_dot_bf16_f32_avx(x, w + (size_t)o * in_dim, in_dim) + (b ? b[o] : 0.0f);
    }
}

int nemo_argmax_bf16_range_avx(const float *x, const uint16_t *w, const float *b,
                               int in_dim, int start, int end, float *best_val_out) {
    int best = start;
    float best_val = -3.4028234663852886e38f;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    int o = start;
    for (; o + 3 < end; o += 4) {
        const uint16_t *w0 = w + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        const uint16_t *w2 = w1 + in_dim;
        const uint16_t *w3 = w2 + in_dim;
        float s0, s1, s2, s3;
        bf16_dot4_avx512(x, w0, w1, w2, w3, in_dim, &s0, &s1, &s2, &s3);
        s0 += b ? b[o] : 0.0f;
        s1 += b ? b[o + 1] : 0.0f;
        s2 += b ? b[o + 2] : 0.0f;
        s3 += b ? b[o + 3] : 0.0f;
        if (s0 > best_val) {
            best_val = s0;
            best = o;
        }
        if (s1 > best_val) {
            best_val = s1;
            best = o + 1;
        }
        if (s2 > best_val) {
            best_val = s2;
            best = o + 2;
        }
        if (s3 > best_val) {
            best_val = s3;
            best = o + 3;
        }
    }
    for (; o + 1 < end; o += 2) {
        const uint16_t *w0 = w + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        float s0, s1;
        bf16_dot2_avx512(x, w0, w1, in_dim, &s0, &s1);
        s0 += b ? b[o] : 0.0f;
        s1 += b ? b[o + 1] : 0.0f;
        if (s0 > best_val) {
            best_val = s0;
            best = o;
        }
        if (s1 > best_val) {
            best_val = s1;
            best = o + 1;
        }
    }
#else
    int o = start;
    for (; o + 1 < end; o += 2) {
        const uint16_t *w0 = w + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        float s0, s1;
        bf16_dot2_avx2(x, w0, w1, in_dim, &s0, &s1);
        s0 += b ? b[o] : 0.0f;
        s1 += b ? b[o + 1] : 0.0f;
        if (s0 > best_val) {
            best_val = s0;
            best = o;
        }
        if (s1 > best_val) {
            best_val = s1;
            best = o + 1;
        }
    }
#endif
    for (; o < end; o++) {
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
