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

static inline float bf16_to_f32_generic(uint16_t x) {
    uint32_t bits = (uint32_t)x << 16;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

float nemo_dot_bf16_f32_generic(const float *a, const uint16_t *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * bf16_to_f32_generic(b[i]);
    return sum;
}

void nemo_bf16_matvec_fused_generic(float *y, const float *x, const uint16_t *w,
                                    const float *b, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        y[o] = nemo_dot_bf16_f32_generic(x, w + (size_t)o * in_dim, in_dim) +
               (b ? b[o] : 0.0f);
    }
}

int nemo_argmax_bf16_range_generic(const float *x, const uint16_t *w, const float *b,
                                   int in_dim, int start, int end, float *best_val_out) {
    int best = start;
    float best_val = -3.4028234663852886e38f;
    for (int o = start; o < end; o++) {
        float v = nemo_dot_bf16_f32_generic(x, w + (size_t)o * in_dim, in_dim) +
                  (b ? b[o] : 0.0f);
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
