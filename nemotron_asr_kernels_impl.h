/*
 * nemotron_asr_kernels_impl.h - architecture dispatch for hot kernels
 */

#ifndef NEMOTRON_ASR_KERNELS_IMPL_H
#define NEMOTRON_ASR_KERNELS_IMPL_H

#include <stdint.h>

float nemo_dot_f32_generic(const float *a, const float *b, int n);
float nemo_dot_bf16_f32_generic(const float *a, const uint16_t *b, int n);
void nemo_bf16_matvec_fused_generic(float *y, const float *x, const uint16_t *w,
                                    const float *b, int in_dim, int out_dim);
int nemo_argmax_bf16_range_generic(const float *x, const uint16_t *w, const float *b,
                                   int in_dim, int start, int end, float *best_val_out);
void nemo_q8p_matvec_fused_generic(float *y, const int8_t *x_q8, float x_scale,
                                   const int8_t *w, const float *w_scales,
                                   const float *b, int stride, int start, int out_dim);
int nemo_argmax_q8p_range_generic(const int8_t *x_q8, float x_scale,
                                  const int8_t *w, const float *w_scales,
                                  const float *b, int stride, int start, int end,
                                  float *best_val_out);
float nemo_attention_score_f32_generic(const float *q, const float *bias_u, const float *k,
                                       const float *bias_v, const float *p, int n);
void nemo_matvec_f32_generic(float *y, const float *x, const float *w, const float *b,
                             int in_dim, int out_dim);
int nemo_argmax_matvec_f32_generic(const float *x, const float *w, const float *b,
                                   int in_dim, int out_dim, float *best_val_out);
void nemo_vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n);
void nemo_fft512_power_f32_generic(float *power, const float *frame);
void nemo_preconv_emit_f32_generic(float *out, const float *history, const float *w, const float *b,
                                   int out_start, int out_t, int total_t,
                                   int c_in, int c_out, int f_in, int f_out,
                                   int k, int stride, int left, int groups);

#ifdef NEMO_FORCE_GENERIC

#define nemo_dot_f32_impl nemo_dot_f32_generic
#define nemo_dot_bf16_f32_impl nemo_dot_bf16_f32_generic
#define nemo_bf16_matvec_fused_impl nemo_bf16_matvec_fused_generic
#define nemo_argmax_bf16_range_impl nemo_argmax_bf16_range_generic
#define nemo_q8p_matvec_fused_impl nemo_q8p_matvec_fused_generic
#define nemo_argmax_q8p_range_impl nemo_argmax_q8p_range_generic
#define nemo_attention_score_f32_impl nemo_attention_score_f32_generic
#define nemo_matvec_f32_impl nemo_matvec_f32_generic
#define nemo_argmax_matvec_f32_impl nemo_argmax_matvec_f32_generic
#define nemo_vec_axpy_inplace_impl nemo_vec_axpy_inplace_generic
#define nemo_fft512_power_f32_impl nemo_fft512_power_f32_generic
#define nemo_preconv_emit_f32_impl nemo_preconv_emit_f32_generic

#elif defined(__ARM_NEON)
float nemo_dot_f32_neon(const float *a, const float *b, int n);
float nemo_dot_bf16_f32_neon(const float *a, const uint16_t *b, int n);
void nemo_bf16_matvec_fused_neon(float *y, const float *x, const uint16_t *w,
                                 const float *b, int in_dim, int out_dim);
int nemo_argmax_bf16_range_neon(const float *x, const uint16_t *w, const float *b,
                                int in_dim, int start, int end, float *best_val_out);
void nemo_q8p_matvec_fused_neon(float *y, const int8_t *x_q8, float x_scale,
                                const int8_t *w, const float *w_scales,
                                const float *b, int stride, int start, int out_dim);
int nemo_argmax_q8p_range_neon(const int8_t *x_q8, float x_scale,
                               const int8_t *w, const float *w_scales,
                               const float *b, int stride, int start, int end,
                               float *best_val_out);
float nemo_attention_score_f32_neon(const float *q, const float *bias_u, const float *k,
                                    const float *bias_v, const float *p, int n);
void nemo_matvec_f32_neon(float *y, const float *x, const float *w, const float *b,
                          int in_dim, int out_dim);
int nemo_argmax_matvec_f32_neon(const float *x, const float *w, const float *b,
                                int in_dim, int out_dim, float *best_val_out);
void nemo_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n);
void nemo_fft512_power_f32_neon(float *power, const float *frame);
void nemo_preconv_emit_f32_neon(float *out, const float *history, const float *w, const float *b,
                                int out_start, int out_t, int total_t,
                                int c_in, int c_out, int f_in, int f_out,
                                int k, int stride, int left, int groups);

#define nemo_dot_f32_impl nemo_dot_f32_neon
#define nemo_dot_bf16_f32_impl nemo_dot_bf16_f32_neon
#define nemo_bf16_matvec_fused_impl nemo_bf16_matvec_fused_neon
#define nemo_argmax_bf16_range_impl nemo_argmax_bf16_range_neon
#define nemo_q8p_matvec_fused_impl nemo_q8p_matvec_fused_neon
#define nemo_argmax_q8p_range_impl nemo_argmax_q8p_range_neon
#define nemo_attention_score_f32_impl nemo_attention_score_f32_neon
#define nemo_matvec_f32_impl nemo_matvec_f32_neon
#define nemo_argmax_matvec_f32_impl nemo_argmax_matvec_f32_neon
#define nemo_vec_axpy_inplace_impl nemo_vec_axpy_inplace_neon
#define nemo_fft512_power_f32_impl nemo_fft512_power_f32_neon
#define nemo_preconv_emit_f32_impl nemo_preconv_emit_f32_neon

#elif defined(__AVX2__) && defined(__FMA__)
float nemo_dot_f32_avx(const float *a, const float *b, int n);
float nemo_dot_bf16_f32_avx(const float *a, const uint16_t *b, int n);
void nemo_bf16_matvec_fused_avx(float *y, const float *x, const uint16_t *w,
                                const float *b, int in_dim, int out_dim);
int nemo_argmax_bf16_range_avx(const float *x, const uint16_t *w, const float *b,
                               int in_dim, int start, int end, float *best_val_out);
void nemo_q8p_matvec_fused_avx(float *y, const int8_t *x_q8, float x_scale,
                               const int8_t *w, const float *w_scales,
                               const float *b, int stride, int start, int out_dim);
int nemo_argmax_q8p_range_avx(const int8_t *x_q8, float x_scale,
                              const int8_t *w, const float *w_scales,
                              const float *b, int stride, int start, int end,
                              float *best_val_out);
float nemo_attention_score_f32_avx(const float *q, const float *bias_u, const float *k,
                                   const float *bias_v, const float *p, int n);
void nemo_matvec_f32_avx(float *y, const float *x, const float *w, const float *b,
                         int in_dim, int out_dim);
int nemo_argmax_matvec_f32_avx(const float *x, const float *w, const float *b,
                               int in_dim, int out_dim, float *best_val_out);
void nemo_vec_axpy_inplace_avx(float *dst, const float *src, float alpha, int n);
void nemo_fft512_power_f32_avx(float *power, const float *frame);
void nemo_preconv_emit_f32_avx(float *out, const float *history, const float *w, const float *b,
                               int out_start, int out_t, int total_t,
                               int c_in, int c_out, int f_in, int f_out,
                               int k, int stride, int left, int groups);

#define nemo_dot_f32_impl nemo_dot_f32_avx
#define nemo_dot_bf16_f32_impl nemo_dot_bf16_f32_avx
#define nemo_bf16_matvec_fused_impl nemo_bf16_matvec_fused_avx
#define nemo_argmax_bf16_range_impl nemo_argmax_bf16_range_avx
#define nemo_q8p_matvec_fused_impl nemo_q8p_matvec_fused_avx
#define nemo_argmax_q8p_range_impl nemo_argmax_q8p_range_avx
#define nemo_attention_score_f32_impl nemo_attention_score_f32_avx
#define nemo_matvec_f32_impl nemo_matvec_f32_avx
#define nemo_argmax_matvec_f32_impl nemo_argmax_matvec_f32_avx
#define nemo_vec_axpy_inplace_impl nemo_vec_axpy_inplace_avx
#define nemo_fft512_power_f32_impl nemo_fft512_power_f32_avx
#define nemo_preconv_emit_f32_impl nemo_preconv_emit_f32_avx

#else
#define nemo_dot_f32_impl nemo_dot_f32_generic
#define nemo_dot_bf16_f32_impl nemo_dot_bf16_f32_generic
#define nemo_bf16_matvec_fused_impl nemo_bf16_matvec_fused_generic
#define nemo_argmax_bf16_range_impl nemo_argmax_bf16_range_generic
#define nemo_q8p_matvec_fused_impl nemo_q8p_matvec_fused_generic
#define nemo_argmax_q8p_range_impl nemo_argmax_q8p_range_generic
#define nemo_attention_score_f32_impl nemo_attention_score_f32_generic
#define nemo_matvec_f32_impl nemo_matvec_f32_generic
#define nemo_argmax_matvec_f32_impl nemo_argmax_matvec_f32_generic
#define nemo_vec_axpy_inplace_impl nemo_vec_axpy_inplace_generic
#define nemo_fft512_power_f32_impl nemo_fft512_power_f32_generic
#define nemo_preconv_emit_f32_impl nemo_preconv_emit_f32_generic
#endif

#endif
