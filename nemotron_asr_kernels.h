/*
 * nemotron_asr_kernels.h - low-level math kernels
 */

#ifndef NEMOTRON_ASR_KERNELS_H
#define NEMOTRON_ASR_KERNELS_H

#include <stddef.h>

float *nemo_alloc(size_t count, size_t elem);
double nemo_time_ms(void);

float nemo_dot_f32(const float *a, const float *b, int n);
float nemo_attention_score_f32(const float *q, const float *bias_u, const float *k,
                               const float *bias_v, const float *p, int n);
void nemo_matvec_f32(float *y, const float *x, const float *w, const float *b,
                     int in_dim, int out_dim);
int nemo_argmax_matvec_f32(const float *x, const float *w, const float *b,
                           int in_dim, int out_dim, float *best_val_out);
void nemo_fft512_power_f32(float *power, const float *frame);

void nemo_conv2d(float *out, const float *in, const float *w, const float *b,
                 int c_in, int c_out, int t_in, int f_in,
                 int k, int stride, int left, int right, int groups);
void nemo_conv1d_depthwise_causal(float *out, const float *in, const float *w,
                                  int t, int dim, int k, int left);
void nemo_preconv_emit_f32(float *out, const float *history, const float *w, const float *b,
                           int out_start, int out_t, int total_t,
                           int c_in, int c_out, int f_in, int f_out,
                           int k, int stride, int left, int groups);

void nemo_vec_axpy_inplace(float *dst, const float *src, float alpha, int n);

void nemo_linear(float *y, const float *x, const float *w, const float *b,
                 int rows, int in_dim, int out_dim);
void nemo_linear_nobias(float *y, const float *x, const float *w,
                        int rows, int in_dim, int out_dim);
void nemo_layer_norm(float *y, const float *x, const float *w, const float *b,
                     int rows, int dim, float eps);
void nemo_softmax(float *x, int n);
void nemo_swish(float *x, int n);
void nemo_relu(float *x, int n);
float nemo_sigmoid(float x);

#endif
