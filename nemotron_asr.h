/*
 * nemotron_asr.h - Nemotron 3.5 ASR Streaming pure C runtime
 */

#ifndef NEMOTRON_ASR_H
#define NEMOTRON_ASR_H

#include <stddef.h>
#include <stdint.h>

#include "nemotron_asr_kernels.h"

#define NEMO_SAMPLE_RATE       16000
#define NEMO_MEL_BINS          128
#define NEMO_N_FFT             512
#define NEMO_WIN_LENGTH        400
#define NEMO_HOP_LENGTH        160

#define NEMO_ENC_LAYERS        24
#define NEMO_D_MODEL           1024
#define NEMO_FFN_DIM           4096
#define NEMO_ENC_HEADS         8
#define NEMO_HEAD_DIM          128
#define NEMO_CONV_CHANNELS     256
#define NEMO_SUBSAMPLING_FACTOR 8
#define NEMO_SUB_FREQ          17
#define NEMO_PREENC_IN         (NEMO_CONV_CHANNELS * NEMO_SUB_FREQ)

#define NEMO_PRED_HIDDEN       640
#define NEMO_PRED_LAYERS       2
#define NEMO_JOINT_HIDDEN      640
#define NEMO_VOCAB_SIZE        13087
#define NEMO_VOCAB_WITH_BLANK  13088
#define NEMO_BLANK_ID          13087
#define NEMO_NUM_PROMPTS       128

typedef struct {
    const char *name;
    const float *data;
    const uint16_t *data_bf16;
    uint8_t dtype;
    uint32_t ndims;
    uint64_t dims[4];
    uint64_t nbytes;
} nemo_tensor_t;

typedef struct {
    const float *norm_ff1_w, *norm_ff1_b;
    nemo_weight_t ff1_linear1_w, ff1_linear2_w;
    const float *norm_conv_w, *norm_conv_b;
    nemo_weight_t conv_pw1_w, conv_pw2_w;
    const float *conv_dw_w, *conv_norm_w, *conv_norm_b;
    const float *norm_att_w, *norm_att_b;
    const float *pos_bias_u, *pos_bias_v;
    nemo_weight_t att_q_w, att_k_w, att_v_w, att_out_w, att_pos_w;
    const float *norm_ff2_w, *norm_ff2_b;
    nemo_weight_t ff2_linear1_w, ff2_linear2_w;
    const float *norm_out_w, *norm_out_b;
} nemo_enc_layer_t;

typedef struct {
    const float *window;
    const float *mel_fb;

    const float *pre_conv0_w, *pre_conv0_b;
    const float *pre_conv2_w, *pre_conv2_b;
    const float *pre_conv3_w, *pre_conv3_b;
    const float *pre_conv5_w, *pre_conv5_b;
    const float *pre_conv6_w, *pre_conv6_b;
    nemo_weight_t pre_out_w;
    const float *pre_out_b;

    nemo_enc_layer_t layers[NEMO_ENC_LAYERS];

    nemo_weight_t prompt0_w, prompt2_w;
    const float *prompt0_b, *prompt2_b;
} nemo_encoder_t;

typedef struct {
    const float *embed_w;
    nemo_weight_t lstm_w_ih[NEMO_PRED_LAYERS];
    nemo_weight_t lstm_w_hh[NEMO_PRED_LAYERS];
    const float *lstm_b_ih[NEMO_PRED_LAYERS];
    const float *lstm_b_hh[NEMO_PRED_LAYERS];
} nemo_decoder_t;

typedef struct {
    nemo_weight_t pred_w, enc_w, out_w;
    const float *pred_b, *enc_b, *out_b;
} nemo_joint_t;

typedef struct {
    void *map;
    size_t map_size;
    nemo_tensor_t *tensors;
    int n_tensors;
    char **vocab;
    int vocab_size;

    nemo_encoder_t encoder;
    nemo_decoder_t decoder;
    nemo_joint_t joint;
} nemo_model_t;

typedef struct {
    nemo_model_t model;
    char model_path[1024];

    int prompt_id;
    int att_left;
    int att_right;
    int strip_lang_tags;
    int max_symbols_per_step;

    double perf_audio_ms;
    double perf_mel_ms;
    double perf_encoder_ms;
    double perf_decoder_ms;
    int perf_frames;
    int perf_tokens;
} nemo_ctx_t;

typedef struct nemo_rnnt_stream_t nemo_rnnt_stream_t;
typedef struct nemo_encoder_stream_t nemo_encoder_stream_t;
typedef struct nemo_mel_stream_t nemo_mel_stream_t;

typedef int (*nemo_encoder_chunk_cb)(void *user, const float *enc, int enc_frames);
typedef int (*nemo_mel_chunk_cb)(void *user, const float *mel, int mel_frames, int final);

float *nemo_load_wav(const char *path, int *out_n_samples);
float *nemo_mel_spectrogram(const nemo_ctx_t *ctx, const float *samples, int n_samples, int *out_frames);
nemo_mel_stream_t *nemo_mel_stream_create(const nemo_ctx_t *ctx);
int nemo_mel_stream_accept(nemo_mel_stream_t *stream, const float *samples, int n_samples,
                           int final, nemo_mel_chunk_cb cb, void *user);
void nemo_mel_stream_free(nemo_mel_stream_t *stream);

int nemo_model_load(nemo_model_t *model, const char *path);
void nemo_model_free(nemo_model_t *model);
const nemo_tensor_t *nemo_model_find(const nemo_model_t *model, const char *name);

nemo_ctx_t *nemo_load(const char *model_path);
void nemo_free(nemo_ctx_t *ctx);
int nemo_set_language(nemo_ctx_t *ctx, const char *lang);
char *nemo_transcribe(nemo_ctx_t *ctx, const char *wav_path);
char *nemo_transcribe_audio(nemo_ctx_t *ctx, const float *samples, int n_samples);

float *nemo_encoder_forward(nemo_ctx_t *ctx, const float *mel, int mel_frames, int *out_frames);
int nemo_encoder_forward_chunks(nemo_ctx_t *ctx, const float *mel, int mel_frames,
                                nemo_encoder_chunk_cb cb, void *user);
nemo_encoder_stream_t *nemo_encoder_stream_create(nemo_ctx_t *ctx);
int nemo_encoder_stream_accept(nemo_ctx_t *ctx, nemo_encoder_stream_t *stream,
                               const float *mel, int mel_frames, int final,
                               nemo_encoder_chunk_cb cb, void *user);
void nemo_encoder_stream_free(nemo_encoder_stream_t *stream);
char *nemo_rnnt_greedy_decode(nemo_ctx_t *ctx, const float *enc, int enc_frames);
nemo_rnnt_stream_t *nemo_rnnt_stream_create(nemo_ctx_t *ctx);
int nemo_rnnt_stream_accept(nemo_ctx_t *ctx, nemo_rnnt_stream_t *stream,
                            const float *enc, int enc_frames);
const char *nemo_rnnt_stream_text(const nemo_rnnt_stream_t *stream);
size_t nemo_rnnt_stream_text_len(const nemo_rnnt_stream_t *stream);
char *nemo_rnnt_stream_finish(nemo_rnnt_stream_t *stream);
void nemo_rnnt_stream_free(nemo_rnnt_stream_t *stream);

#endif
