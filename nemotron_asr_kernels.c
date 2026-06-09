#include "nemotron_asr.h"
#include "nemotron_asr_kernels_impl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

float *nemo_alloc(size_t count, size_t elem) {
    if (count == 0 || elem == 0 || count > (SIZE_MAX / elem)) return NULL;
    float *p = (float *)calloc(count, elem);
    if (!p) {
        fprintf(stderr, "nemotron: allocation failed (%zu x %zu)\n", count, elem);
    }
    return p;
}

double nemo_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

void nemo_linear(float *y, const float *x, const float *w, const float *b,
                 int rows, int in_dim, int out_dim) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * in_dim;
        float *yr = y + (size_t)r * out_dim;
        nemo_matvec_f32_impl(yr, xr, w, b, in_dim, out_dim);
    }
}

void nemo_linear_nobias(float *y, const float *x, const float *w,
                        int rows, int in_dim, int out_dim) {
    nemo_linear(y, x, w, NULL, rows, in_dim, out_dim);
}

static int conv_out_len(int n, int left, int right, int k, int stride) {
    return (n + left + right - k) / stride + 1;
}

void nemo_conv2d(float *out, const float *in, const float *w, const float *b,
                 int c_in, int c_out, int t_in, int f_in,
                 int k, int stride, int left, int right, int groups) {
    int t_out = conv_out_len(t_in, left, right, k, stride);
    int f_out = conv_out_len(f_in, left, right, k, stride);
    int in_per_group = c_in / groups;
    int out_per_group = c_out / groups;
    for (int oc = 0; oc < c_out; oc++) {
        int g = oc / out_per_group;
        int ic0 = g * in_per_group;
        for (int ot = 0; ot < t_out; ot++) {
            for (int of = 0; of < f_out; of++) {
                float sum = b ? b[oc] : 0.0f;
                for (int icg = 0; icg < in_per_group; icg++) {
                    int ic = ic0 + icg;
                    int w_ic = (groups == c_in && in_per_group == 1) ? 0 : icg;
                    for (int kt = 0; kt < k; kt++) {
                        int it = ot * stride + kt - left;
                        if (it < 0 || it >= t_in) continue;
                        for (int kf = 0; kf < k; kf++) {
                            int iff = of * stride + kf - left;
                            if (iff < 0 || iff >= f_in) continue;
                            size_t iidx = ((size_t)ic * t_in + it) * f_in + iff;
                            size_t widx = (((size_t)oc * in_per_group + w_ic) * k + kt) * k + kf;
                            sum += in[iidx] * w[widx];
                        }
                    }
                }
                out[((size_t)oc * t_out + ot) * f_out + of] = sum;
            }
        }
    }
}

void nemo_conv1d_depthwise_causal(float *out, const float *in, const float *w,
                                  int t, int dim, int k, int left) {
    for (int tt = 0; tt < t; tt++) {
        for (int c = 0; c < dim; c++) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; kk++) {
                int src = tt + kk - left;
                if (src >= 0 && src < t) {
                    sum += in[(size_t)src * dim + c] * w[(size_t)c * k + kk];
                }
            }
            out[(size_t)tt * dim + c] = sum;
        }
    }
}

void nemo_preconv_emit_f32(float *out, const float *history, const float *w, const float *b,
                           int out_start, int out_t, int total_t,
                           int c_in, int c_out, int f_in, int f_out,
                           int k, int stride, int left, int groups) {
    nemo_preconv_emit_f32_impl(out, history, w, b, out_start, out_t, total_t,
                               c_in, c_out, f_in, f_out, k, stride, left, groups);
}

float nemo_dot_f32(const float *a, const float *b, int n) {
    return nemo_dot_f32_impl(a, b, n);
}

float nemo_attention_score_f32(const float *q, const float *bias_u, const float *k,
                               const float *bias_v, const float *p, int n) {
    return nemo_attention_score_f32_impl(q, bias_u, k, bias_v, p, n);
}

void nemo_matvec_f32(float *y, const float *x, const float *w, const float *b,
                     int in_dim, int out_dim) {
    nemo_matvec_f32_impl(y, x, w, b, in_dim, out_dim);
}

int nemo_argmax_matvec_f32(const float *x, const float *w, const float *b,
                           int in_dim, int out_dim, float *best_val_out) {
    return nemo_argmax_matvec_f32_impl(x, w, b, in_dim, out_dim, best_val_out);
}

void nemo_fft512_power_f32(float *power, const float *frame) {
    nemo_fft512_power_f32_impl(power, frame);
}

void nemo_vec_axpy_inplace(float *dst, const float *src, float alpha, int n) {
    nemo_vec_axpy_inplace_impl(dst, src, alpha, n);
}

void nemo_layer_norm(float *y, const float *x, const float *w, const float *b,
                     int rows, int dim, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *yr = y + (size_t)r * dim;
        double mean = 0.0;
        for (int i = 0; i < dim; i++) mean += xr[i];
        mean /= (double)dim;
        double var = 0.0;
        for (int i = 0; i < dim; i++) {
            double d = (double)xr[i] - mean;
            var += d * d;
        }
        float inv = 1.0f / sqrtf((float)(var / (double)dim) + eps);
        for (int i = 0; i < dim; i++) {
            float v = (xr[i] - (float)mean) * inv;
            yr[i] = v * w[i] + b[i];
        }
    }
}

void nemo_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    float inv = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

float nemo_sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

void nemo_swish(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] * nemo_sigmoid(x[i]);
}

void nemo_relu(float *x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}
