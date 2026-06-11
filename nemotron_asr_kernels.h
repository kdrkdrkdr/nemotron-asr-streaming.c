/*
 * nemotron_asr_kernels.h - low-level math kernels
 */

#ifndef NEMOTRON_ASR_KERNELS_H
#define NEMOTRON_ASR_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#define NEMO_TENSOR_F32  1u
#define NEMO_TENSOR_BF16 2u
#define NEMO_TENSOR_Q8   3u
#define NEMO_TENSOR_Q8P  4u

typedef struct {
    const float *f32;
    const uint16_t *bf16;
    const int8_t *q8;
    const float *q8_scales;
    uint32_t q8_stride;
    uint8_t q8_packed;
    uint8_t dtype;
} nemo_weight_t;

float *nemo_alloc(size_t count, size_t elem);
double nemo_time_ms(void);
int nemo_get_num_cpus(void);
void nemo_set_threads(int n_threads);
int nemo_get_threads(void);

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
void nemo_linear_weight(float *y, const float *x, const nemo_weight_t *w, const float *b,
                        int rows, int in_dim, int out_dim);
void nemo_linear_nobias_weight(float *y, const float *x, const nemo_weight_t *w,
                               int rows, int in_dim, int out_dim);
void nemo_linear3_nobias(float *y0, float *y1, float *y2, const float *x,
                         const float *w0, const float *w1, const float *w2,
                         int rows, int in_dim, int out_dim);
void nemo_linear3_nobias_weight(float *y0, float *y1, float *y2, const float *x,
                                const nemo_weight_t *w0, const nemo_weight_t *w1,
                                const nemo_weight_t *w2,
                                int rows, int in_dim, int out_dim);
void nemo_prompt_linear_relu(float *y, const float *x, const float *w, const float *b,
                             int rows, int in_dim, int prompt_dim, int prompt_id,
                             int out_dim);
void nemo_prompt_linear_relu_weight(float *y, const float *x, const nemo_weight_t *w,
                                    const float *b, int rows, int in_dim,
                                    int prompt_dim, int prompt_id, int out_dim);
void nemo_lstm_gates_f32(float *y, const float *x, const float *h,
                         const float *w_ih, const float *w_hh,
                         const float *b_ih, const float *b_hh,
                         int dim, int out_dim);
void nemo_lstm_gates_weight(float *y, const float *x, const float *h,
                            const nemo_weight_t *w_ih, const nemo_weight_t *w_hh,
                            const float *b_ih, const float *b_hh,
                            int dim, int out_dim);
int nemo_argmax_matvec_weight(const float *x, const nemo_weight_t *w, const float *b,
                              int in_dim, int out_dim, float *best_val_out);
void nemo_layer_norm(float *y, const float *x, const float *w, const float *b,
                     int rows, int dim, float eps);
void nemo_softmax(float *x, int n);
void nemo_swish(float *x, int n);
void nemo_relu(float *x, int n);
float nemo_sigmoid(float x);

#endif
