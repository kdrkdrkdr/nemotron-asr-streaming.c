#include "nemotron_asr.h"

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
        for (int o = 0; o < out_dim; o++) {
            const float *wr = w + (size_t)o * in_dim;
            float sum = b ? b[o] : 0.0f;
            for (int i = 0; i < in_dim; i++) sum += xr[i] * wr[i];
            yr[o] = sum;
        }
    }
}

void nemo_linear_nobias(float *y, const float *x, const float *w,
                        int rows, int in_dim, int out_dim) {
    nemo_linear(y, x, w, NULL, rows, in_dim, out_dim);
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
