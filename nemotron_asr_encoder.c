#include "nemotron_asr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LN_EPS 1.0e-5f
#define NEG_INF -10000.0f
#define CONV_LEFT 8

typedef struct {
    int att_cap;
    int conv_cap;
    int pos_min;
    int pos_len;
    int att_len[NEMO_ENC_LAYERS];
    int conv_len[NEMO_ENC_LAYERS];
    float *att_k[NEMO_ENC_LAYERS];
    float *att_v[NEMO_ENC_LAYERS];
    float *conv_glu[NEMO_ENC_LAYERS];
    float *rel_pos[NEMO_ENC_LAYERS];
} enc_stream_state_t;

typedef struct {
    int c_in, c_out;
    int f_in, f_out;
    int k, stride, left, right, groups;
    int total_t, cap_t, next_out;
    float *buf; /* [time, channel, freq] */
} pre_conv_stage_t;

typedef struct {
    pre_conv_stage_t c0;
    pre_conv_stage_t c2;
    pre_conv_stage_t c3;
    pre_conv_stage_t c5;
    pre_conv_stage_t c6;
} preenc_stream_state_t;

struct nemo_encoder_stream_t {
    enc_stream_state_t enc;
    preenc_stream_state_t pre;
    float *pending;
    int pending_len;
    int pending_cap;
    int chunk;
};

static void make_rel_pe_range(float *pe, int min_pos, int max_pos);
static int conv_out_len(int n, int left, int right, int k, int stride);

static void enc_stream_free(enc_stream_state_t *s) {
    if (!s) return;
    for (int i = 0; i < NEMO_ENC_LAYERS; i++) {
        free(s->att_k[i]);
        free(s->att_v[i]);
        free(s->conv_glu[i]);
        free(s->rel_pos[i]);
    }
    memset(s, 0, sizeof(*s));
}

static int enc_stream_init(enc_stream_state_t *s, int att_cap, int conv_cap) {
    memset(s, 0, sizeof(*s));
    s->att_cap = att_cap;
    s->conv_cap = conv_cap;
    for (int i = 0; i < NEMO_ENC_LAYERS; i++) {
        s->att_k[i] = nemo_alloc((size_t)att_cap * NEMO_D_MODEL, sizeof(float));
        s->att_v[i] = nemo_alloc((size_t)att_cap * NEMO_D_MODEL, sizeof(float));
        s->conv_glu[i] = nemo_alloc((size_t)conv_cap * NEMO_D_MODEL, sizeof(float));
        if (!s->att_k[i] || !s->att_v[i] || !s->conv_glu[i]) {
            enc_stream_free(s);
            return -1;
        }
    }
    return 0;
}

static int enc_stream_build_rel_pos(const nemo_ctx_t *ctx, enc_stream_state_t *s, int chunk) {
    s->pos_min = -(chunk - 1);
    int pos_max = s->att_cap + chunk - 1;
    s->pos_len = pos_max - s->pos_min + 1;
    float *pe = nemo_alloc((size_t)s->pos_len * NEMO_D_MODEL, sizeof(float));
    if (!pe) return -1;
    make_rel_pe_range(pe, s->pos_min, pos_max);
    for (int i = 0; i < NEMO_ENC_LAYERS; i++) {
        s->rel_pos[i] = nemo_alloc((size_t)s->pos_len * NEMO_D_MODEL, sizeof(float));
        if (!s->rel_pos[i]) {
            free(pe);
            return -1;
        }
        nemo_linear_nobias(s->rel_pos[i], pe, ctx->model.encoder.layers[i].att_pos_w,
                           s->pos_len, NEMO_D_MODEL, NEMO_D_MODEL);
    }
    free(pe);
    return 0;
}

static void cache_append(float *cache, int *len, int cap, const float *src, int n, int dim) {
    if (cap <= 0 || n <= 0) return;
    if (n >= cap) {
        memcpy(cache, src + (size_t)(n - cap) * dim, (size_t)cap * dim * sizeof(float));
        *len = cap;
        return;
    }
    int keep = *len;
    if (keep + n > cap) keep = cap - n;
    if (keep > 0 && keep != *len) {
        memmove(cache, cache + (size_t)(*len - keep) * dim, (size_t)keep * dim * sizeof(float));
    }
    memcpy(cache + (size_t)keep * dim, src, (size_t)n * dim * sizeof(float));
    *len = keep + n;
}

static int conv_out_len(int n, int left, int right, int k, int stride) {
    return (n + left + right - k) / stride + 1;
}

static void pre_stage_init(pre_conv_stage_t *s, int c_in, int c_out, int f_in,
                           int k, int stride, int left, int right, int groups) {
    memset(s, 0, sizeof(*s));
    s->c_in = c_in;
    s->c_out = c_out;
    s->f_in = f_in;
    s->f_out = conv_out_len(f_in, left, right, k, stride);
    s->k = k;
    s->stride = stride;
    s->left = left;
    s->right = right;
    s->groups = groups;
}

static void pre_stage_free(pre_conv_stage_t *s) {
    free(s->buf);
    memset(s, 0, sizeof(*s));
}

static int pre_stage_append(pre_conv_stage_t *s, const float *x, int n) {
    if (n <= 0) return 0;
    if (s->total_t + n > s->cap_t) {
        int nc = s->cap_t ? s->cap_t * 2 : 64;
        while (nc < s->total_t + n) nc *= 2;
        float *p = (float *)realloc(s->buf, (size_t)nc * s->c_in * s->f_in * sizeof(float));
        if (!p) return -1;
        s->buf = p;
        s->cap_t = nc;
    }
    memcpy(s->buf + (size_t)s->total_t * s->c_in * s->f_in,
           x, (size_t)n * s->c_in * s->f_in * sizeof(float));
    s->total_t += n;
    return 0;
}

static int pre_stage_available(const pre_conv_stage_t *s, int final) {
    if (s->total_t <= 0) return 0;
    if (final) return conv_out_len(s->total_t, s->left, s->right, s->k, s->stride);
    int need = (s->k - 1) - s->left;
    int last = s->total_t - 1 - need;
    if (last < 0) return 0;
    return last / s->stride + 1;
}

static int pre_stage_accept(pre_conv_stage_t *s, const float *x, int n, int final,
                            const float *w, const float *b, float **out, int *out_t) {
    *out = NULL;
    *out_t = 0;
    if (pre_stage_append(s, x, n) != 0) return -1;
    int avail = pre_stage_available(s, final);
    int new_t = avail - s->next_out;
    if (new_t <= 0) return 0;
    float *y = nemo_alloc((size_t)new_t * s->c_out * s->f_out, sizeof(float));
    if (!y) return -1;
    nemo_preconv_emit_f32(y, s->buf, w, b, s->next_out, new_t, s->total_t,
                          s->c_in, s->c_out, s->f_in, s->f_out,
                          s->k, s->stride, s->left, s->groups);
    s->next_out = avail;
    *out = y;
    *out_t = new_t;
    return 0;
}

static void preenc_stream_init(preenc_stream_state_t *s) {
    pre_stage_init(&s->c0, 1, NEMO_CONV_CHANNELS, NEMO_MEL_BINS, 3, 2, 2, 1, 1);
    pre_stage_init(&s->c2, NEMO_CONV_CHANNELS, NEMO_CONV_CHANNELS, s->c0.f_out, 3, 2, 2, 1, NEMO_CONV_CHANNELS);
    pre_stage_init(&s->c3, NEMO_CONV_CHANNELS, NEMO_CONV_CHANNELS, s->c2.f_out, 1, 1, 0, 0, 1);
    pre_stage_init(&s->c5, NEMO_CONV_CHANNELS, NEMO_CONV_CHANNELS, s->c3.f_out, 3, 2, 2, 1, NEMO_CONV_CHANNELS);
    pre_stage_init(&s->c6, NEMO_CONV_CHANNELS, NEMO_CONV_CHANNELS, s->c5.f_out, 1, 1, 0, 0, 1);
}

static void preenc_stream_free(preenc_stream_state_t *s) {
    pre_stage_free(&s->c0);
    pre_stage_free(&s->c2);
    pre_stage_free(&s->c3);
    pre_stage_free(&s->c5);
    pre_stage_free(&s->c6);
}

static void relu_buf(float *x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}

static int preenc_stream_accept(const nemo_ctx_t *ctx, preenc_stream_state_t *s,
                                const float *mel, int mel_start, int mel_frames,
                                int mel_stride, int final, float **out, int *out_t) {
    const nemo_encoder_t *e = &ctx->model.encoder;
    *out = NULL;
    *out_t = 0;
    float *x0 = NULL;
    if (mel_frames > 0) {
        x0 = nemo_alloc((size_t)mel_frames * NEMO_MEL_BINS, sizeof(float));
        if (!x0) return -1;
        for (int t = 0; t < mel_frames; t++) {
            for (int f = 0; f < NEMO_MEL_BINS; f++) {
                x0[(size_t)t * NEMO_MEL_BINS + f] = mel[(size_t)f * mel_stride + mel_start + t];
            }
        }
    }

    float *x1 = NULL, *x2 = NULL, *x3 = NULL, *x4 = NULL, *x5 = NULL;
    int t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
    if (pre_stage_accept(&s->c0, x0, mel_frames, final, e->pre_conv0_w, e->pre_conv0_b, &x1, &t1) != 0) goto fail;
    free(x0); x0 = NULL;
    if (x1) relu_buf(x1, t1 * NEMO_CONV_CHANNELS * s->c0.f_out);
    if (pre_stage_accept(&s->c2, x1, t1, final, e->pre_conv2_w, e->pre_conv2_b, &x2, &t2) != 0) goto fail;
    free(x1); x1 = NULL;
    if (pre_stage_accept(&s->c3, x2, t2, final, e->pre_conv3_w, e->pre_conv3_b, &x3, &t3) != 0) goto fail;
    free(x2); x2 = NULL;
    if (x3) relu_buf(x3, t3 * NEMO_CONV_CHANNELS * s->c3.f_out);
    if (pre_stage_accept(&s->c5, x3, t3, final, e->pre_conv5_w, e->pre_conv5_b, &x4, &t4) != 0) goto fail;
    free(x3); x3 = NULL;
    if (pre_stage_accept(&s->c6, x4, t4, final, e->pre_conv6_w, e->pre_conv6_b, &x5, &t5) != 0) goto fail;
    free(x4); x4 = NULL;
    if (x5) relu_buf(x5, t5 * NEMO_CONV_CHANNELS * s->c6.f_out);
    if (t5 <= 0) { free(x5); return 0; }
    if (s->c6.f_out != NEMO_SUB_FREQ) {
        fprintf(stderr, "nemotron: unexpected subsampling frequency %d (expected %d)\n", s->c6.f_out, NEMO_SUB_FREQ);
        free(x5);
        return -1;
    }
    float *flat = nemo_alloc((size_t)t5 * NEMO_PREENC_IN, sizeof(float));
    float *proj = nemo_alloc((size_t)t5 * NEMO_D_MODEL, sizeof(float));
    if (!flat || !proj) { free(x5); free(flat); free(proj); return -1; }
    for (int t = 0; t < t5; t++) {
        for (int c = 0; c < NEMO_CONV_CHANNELS; c++) {
            for (int f = 0; f < NEMO_SUB_FREQ; f++) {
                flat[(size_t)t * NEMO_PREENC_IN + c * NEMO_SUB_FREQ + f] =
                    x5[((size_t)t * NEMO_CONV_CHANNELS + c) * NEMO_SUB_FREQ + f];
            }
        }
    }
    free(x5);
    nemo_linear(proj, flat, e->pre_out_w, e->pre_out_b, t5, NEMO_PREENC_IN, NEMO_D_MODEL);
    free(flat);
    *out = proj;
    *out_t = t5;
    return 0;
fail:
    free(x0); free(x1); free(x2); free(x3); free(x4); free(x5);
    return -1;
}

static void make_rel_pe_range(float *pe, int min_pos, int max_pos) {
    int pos_len = max_pos - min_pos + 1;
    for (int r = 0; r < pos_len; r++) {
        float pos = (float)(min_pos + r);
        for (int i = 0; i < NEMO_D_MODEL / 2; i++) {
            float div = expf(-(logf(10000.0f) / (float)NEMO_D_MODEL) * (float)(2 * i));
            pe[(size_t)r * NEMO_D_MODEL + 2 * i] = sinf(pos * div);
            pe[(size_t)r * NEMO_D_MODEL + 2 * i + 1] = cosf(pos * div);
        }
    }
}

static int rel_attention_stream(const nemo_ctx_t *ctx, const nemo_enc_layer_t *l,
                                enc_stream_state_t *s, int layer, const float *x,
                                int t, float *out) {
    (void)ctx;
    int cache_len = s->att_len[layer];
    int total_k = cache_len + t;
    float *q = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    float *k_cur = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    float *v_cur = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    float *ctxv = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    float *scores = nemo_alloc((size_t)total_k, sizeof(float));
    const float *p = s->rel_pos[layer];
    if (!q || !k_cur || !v_cur || !ctxv || !scores || !p) {
        free(q); free(k_cur); free(v_cur); free(ctxv); free(scores);
        return -1;
    }
    nemo_linear_nobias(q, x, l->att_q_w, t, NEMO_D_MODEL, NEMO_D_MODEL);
    nemo_linear_nobias(k_cur, x, l->att_k_w, t, NEMO_D_MODEL, NEMO_D_MODEL);
    nemo_linear_nobias(v_cur, x, l->att_v_w, t, NEMO_D_MODEL, NEMO_D_MODEL);

    const float scale = 1.0f / sqrtf((float)NEMO_HEAD_DIM);
    for (int qi = 0; qi < t; qi++) {
        for (int h = 0; h < NEMO_ENC_HEADS; h++) {
            const float *qh = q + (size_t)qi * NEMO_D_MODEL + h * NEMO_HEAD_DIM;
            const float *bu = l->pos_bias_u + h * NEMO_HEAD_DIM;
            const float *bv = l->pos_bias_v + h * NEMO_HEAD_DIM;
            for (int kj = 0; kj < total_k; kj++) {
                int key_local = kj - cache_len;
                int rel_pos = qi - key_local;
                const float *kh;
                if (kj < cache_len) {
                    kh = s->att_k[layer] + (size_t)kj * NEMO_D_MODEL + h * NEMO_HEAD_DIM;
                } else {
                    kh = k_cur + (size_t)(kj - cache_len) * NEMO_D_MODEL + h * NEMO_HEAD_DIM;
                }
                const float *ph = p + (size_t)(rel_pos - s->pos_min) * NEMO_D_MODEL + h * NEMO_HEAD_DIM;
                scores[kj] = nemo_attention_score_f32(qh, bu, kh, bv, ph, NEMO_HEAD_DIM) * scale;
            }
            nemo_softmax(scores, total_k);
            float *dst = ctxv + (size_t)qi * NEMO_D_MODEL + h * NEMO_HEAD_DIM;
            memset(dst, 0, sizeof(float) * NEMO_HEAD_DIM);
            for (int kj = 0; kj < total_k; kj++) {
                const float *vh;
                if (kj < cache_len) {
                    vh = s->att_v[layer] + (size_t)kj * NEMO_D_MODEL + h * NEMO_HEAD_DIM;
                } else {
                    vh = v_cur + (size_t)(kj - cache_len) * NEMO_D_MODEL + h * NEMO_HEAD_DIM;
                }
                nemo_vec_axpy_inplace(dst, vh, scores[kj], NEMO_HEAD_DIM);
            }
        }
    }
    nemo_linear_nobias(out, ctxv, l->att_out_w, t, NEMO_D_MODEL, NEMO_D_MODEL);
    int next_len = s->att_len[layer];
    int next_v_len = s->att_len[layer];
    cache_append(s->att_k[layer], &next_len, s->att_cap, k_cur, t, NEMO_D_MODEL);
    cache_append(s->att_v[layer], &next_v_len, s->att_cap, v_cur, t, NEMO_D_MODEL);
    s->att_len[layer] = next_len;
    free(q); free(k_cur); free(v_cur); free(ctxv); free(scores);
    return 0;
}

static int conformer_conv_stream(const nemo_enc_layer_t *l, enc_stream_state_t *s,
                                 int layer, const float *x, int t, float *out) {
    float *pw = nemo_alloc((size_t)t * 2 * NEMO_D_MODEL, sizeof(float));
    float *glu = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    float *dw = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    float *norm = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    if (!pw || !glu || !dw || !norm) {
        free(pw); free(glu); free(dw); free(norm);
        return -1;
    }
    nemo_linear_nobias(pw, x, l->conv_pw1_w, t, NEMO_D_MODEL, 2 * NEMO_D_MODEL);
    for (int i = 0; i < t; i++) {
        for (int d = 0; d < NEMO_D_MODEL; d++) {
            float a = pw[(size_t)i * 2 * NEMO_D_MODEL + d];
            float b = pw[(size_t)i * 2 * NEMO_D_MODEL + NEMO_D_MODEL + d];
            glu[(size_t)i * NEMO_D_MODEL + d] = a * nemo_sigmoid(b);
        }
    }
    for (int tt = 0; tt < t; tt++) {
        for (int c = 0; c < NEMO_D_MODEL; c++) {
            float sum = 0.0f;
            for (int kk = 0; kk < 9; kk++) {
                int src = tt + kk - CONV_LEFT;
                const float *row = NULL;
                if (src < 0) {
                    int ci = s->conv_len[layer] + src;
                    if (ci >= 0) row = s->conv_glu[layer] + (size_t)ci * NEMO_D_MODEL;
                } else if (src < t) {
                    row = glu + (size_t)src * NEMO_D_MODEL;
                }
                if (row) sum += row[c] * l->conv_dw_w[(size_t)c * 9 + kk];
            }
            dw[(size_t)tt * NEMO_D_MODEL + c] = sum;
        }
    }
    cache_append(s->conv_glu[layer], &s->conv_len[layer], s->conv_cap, glu, t, NEMO_D_MODEL);
    nemo_layer_norm(norm, dw, l->conv_norm_w, l->conv_norm_b, t, NEMO_D_MODEL, LN_EPS);
    nemo_swish(norm, t * NEMO_D_MODEL);
    nemo_linear_nobias(out, norm, l->conv_pw2_w, t, NEMO_D_MODEL, NEMO_D_MODEL);
    free(pw); free(glu); free(dw); free(norm);
    return 0;
}

static int conformer_layer_stream(nemo_ctx_t *ctx, const nemo_enc_layer_t *l,
                                  enc_stream_state_t *s, int layer, float *x, int t) {
    float *norm = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    float *ff = nemo_alloc((size_t)t * NEMO_FFN_DIM, sizeof(float));
    float *tmp = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    if (!norm || !ff || !tmp) { free(norm); free(ff); free(tmp); return -1; }

    nemo_layer_norm(norm, x, l->norm_ff1_w, l->norm_ff1_b, t, NEMO_D_MODEL, LN_EPS);
    nemo_linear_nobias(ff, norm, l->ff1_linear1_w, t, NEMO_D_MODEL, NEMO_FFN_DIM);
    nemo_swish(ff, t * NEMO_FFN_DIM);
    nemo_linear_nobias(tmp, ff, l->ff1_linear2_w, t, NEMO_FFN_DIM, NEMO_D_MODEL);
    nemo_vec_axpy_inplace(x, tmp, 0.5f, t * NEMO_D_MODEL);

    nemo_layer_norm(norm, x, l->norm_att_w, l->norm_att_b, t, NEMO_D_MODEL, LN_EPS);
    if (rel_attention_stream(ctx, l, s, layer, norm, t, tmp) != 0) { free(norm); free(ff); free(tmp); return -1; }
    nemo_vec_axpy_inplace(x, tmp, 1.0f, t * NEMO_D_MODEL);

    nemo_layer_norm(norm, x, l->norm_conv_w, l->norm_conv_b, t, NEMO_D_MODEL, LN_EPS);
    if (conformer_conv_stream(l, s, layer, norm, t, tmp) != 0) { free(norm); free(ff); free(tmp); return -1; }
    nemo_vec_axpy_inplace(x, tmp, 1.0f, t * NEMO_D_MODEL);

    nemo_layer_norm(norm, x, l->norm_ff2_w, l->norm_ff2_b, t, NEMO_D_MODEL, LN_EPS);
    nemo_linear_nobias(ff, norm, l->ff2_linear1_w, t, NEMO_D_MODEL, NEMO_FFN_DIM);
    nemo_swish(ff, t * NEMO_FFN_DIM);
    nemo_linear_nobias(tmp, ff, l->ff2_linear2_w, t, NEMO_FFN_DIM, NEMO_D_MODEL);
    nemo_vec_axpy_inplace(x, tmp, 0.5f, t * NEMO_D_MODEL);

    nemo_layer_norm(tmp, x, l->norm_out_w, l->norm_out_b, t, NEMO_D_MODEL, LN_EPS);
    memcpy(x, tmp, (size_t)t * NEMO_D_MODEL * sizeof(float));
    free(norm); free(ff); free(tmp);
    return 0;
}

static int apply_prompt(const nemo_ctx_t *ctx, float *x, int t) {
    const nemo_encoder_t *e = &ctx->model.encoder;
    float *h = nemo_alloc((size_t)t * 2 * NEMO_D_MODEL, sizeof(float));
    float *y = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    if (!h || !y) { free(h); free(y); return -1; }
    for (int row = 0; row < t; row++) {
        const float *xr = x + (size_t)row * NEMO_D_MODEL;
        float *hr = h + (size_t)row * 2 * NEMO_D_MODEL;
        for (int o = 0; o < 2 * NEMO_D_MODEL; o++) {
            const float *wr = e->prompt0_w + (size_t)o * (NEMO_D_MODEL + NEMO_NUM_PROMPTS);
            float sum = e->prompt0_b[o] + wr[NEMO_D_MODEL + ctx->prompt_id];
            sum += nemo_dot_f32(xr, wr, NEMO_D_MODEL);
            hr[o] = sum > 0.0f ? sum : 0.0f;
        }
    }
    nemo_linear(y, h, e->prompt2_w, e->prompt2_b, t, 2 * NEMO_D_MODEL, NEMO_D_MODEL);
    memcpy(x, y, (size_t)t * NEMO_D_MODEL * sizeof(float));
    free(h); free(y);
    return 0;
}

nemo_encoder_stream_t *nemo_encoder_stream_create(nemo_ctx_t *ctx) {
    nemo_encoder_stream_t *s = (nemo_encoder_stream_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    preenc_stream_init(&s->pre);
    int att_cap = ctx->att_left >= 0 ? ctx->att_left : 56;
    if (enc_stream_init(&s->enc, att_cap, CONV_LEFT) != 0) {
        nemo_encoder_stream_free(s);
        return NULL;
    }

    s->chunk = ctx->att_right + 1;
    if (s->chunk < 1) s->chunk = 1;
    if (enc_stream_build_rel_pos(ctx, &s->enc, s->chunk) != 0) {
        nemo_encoder_stream_free(s);
        return NULL;
    }
    return s;
}

void nemo_encoder_stream_free(nemo_encoder_stream_t *s) {
    if (!s) return;
    enc_stream_free(&s->enc);
    preenc_stream_free(&s->pre);
    free(s->pending);
    free(s);
}

int nemo_encoder_stream_accept(nemo_ctx_t *ctx, nemo_encoder_stream_t *s,
                               const float *mel, int mel_frames, int final,
                               nemo_encoder_chunk_cb cb, void *user) {
    if (!s || !cb || mel_frames < 0) return -1;
    int mel_step = s->chunk * NEMO_SUBSAMPLING_FACTOR;
    if (mel_step < 1) mel_step = NEMO_SUBSAMPLING_FACTOR;
    int passes = mel_frames > 0 ? (mel_frames + mel_step - 1) / mel_step : 1;

    for (int pass = 0; pass < passes; pass++) {
        int mstart = pass * mel_step;
        int mn = mel_frames - mstart;
        if (mn < 0) mn = 0;
        if (mn > mel_step) mn = mel_step;
        int pass_final = final && (pass + 1 == passes);
        float *pre_x = NULL;
        int pre_t = 0;
        if (preenc_stream_accept(ctx, &s->pre, mel, mstart, mn, mel_frames, pass_final, &pre_x, &pre_t) != 0) {
            return -1;
        }
        if (pre_t > 0) {
            if (s->pending_len + pre_t > s->pending_cap) {
                int nc = s->pending_cap ? s->pending_cap * 2 : 32;
                while (nc < s->pending_len + pre_t) nc *= 2;
                float *p = (float *)realloc(s->pending, (size_t)nc * NEMO_D_MODEL * sizeof(float));
                if (!p) {
                    free(pre_x);
                    return -1;
                }
                s->pending = p;
                s->pending_cap = nc;
            }
            memcpy(s->pending + (size_t)s->pending_len * NEMO_D_MODEL,
                   pre_x, (size_t)pre_t * NEMO_D_MODEL * sizeof(float));
            s->pending_len += pre_t;
        }
        free(pre_x);

        while (s->pending_len >= s->chunk || (pass_final && s->pending_len > 0)) {
            int n = s->pending_len < s->chunk ? s->pending_len : s->chunk;
            float *cur = nemo_alloc((size_t)n * NEMO_D_MODEL, sizeof(float));
            if (!cur) {
                return -1;
            }
            memcpy(cur, s->pending, (size_t)n * NEMO_D_MODEL * sizeof(float));
            for (int i = 0; i < NEMO_ENC_LAYERS; i++) {
                if (conformer_layer_stream(ctx, &ctx->model.encoder.layers[i], &s->enc, i, cur, n) != 0) {
                    free(cur);
                    return -1;
                }
            }
            if (apply_prompt(ctx, cur, n) != 0 || cb(user, cur, n) != 0) {
                free(cur);
                return -1;
            }
            free(cur);
            s->pending_len -= n;
            if (s->pending_len > 0) {
                memmove(s->pending, s->pending + (size_t)n * NEMO_D_MODEL,
                        (size_t)s->pending_len * NEMO_D_MODEL * sizeof(float));
            }
        }
    }
    return 0;
}

int nemo_encoder_forward_chunks(nemo_ctx_t *ctx, const float *mel, int mel_frames,
                                nemo_encoder_chunk_cb cb, void *user) {
    nemo_encoder_stream_t *s = nemo_encoder_stream_create(ctx);
    if (!s) return -1;
    int rc = nemo_encoder_stream_accept(ctx, s, mel, mel_frames, 1, cb, user);
    nemo_encoder_stream_free(s);
    return rc;
}

typedef struct {
    float *data;
    int frames;
    int cap;
} enc_collect_t;

static int collect_encoder_chunk(void *user, const float *enc, int enc_frames) {
    enc_collect_t *c = (enc_collect_t *)user;
    if (c->frames + enc_frames > c->cap) {
        int nc = c->cap ? c->cap * 2 : 32;
        while (nc < c->frames + enc_frames) nc *= 2;
        float *p = (float *)realloc(c->data, (size_t)nc * NEMO_D_MODEL * sizeof(float));
        if (!p) return -1;
        c->data = p;
        c->cap = nc;
    }
    memcpy(c->data + (size_t)c->frames * NEMO_D_MODEL, enc,
           (size_t)enc_frames * NEMO_D_MODEL * sizeof(float));
    c->frames += enc_frames;
    return 0;
}

float *nemo_encoder_forward(nemo_ctx_t *ctx, const float *mel, int mel_frames, int *out_frames) {
    enc_collect_t c = {0};
    if (nemo_encoder_forward_chunks(ctx, mel, mel_frames, collect_encoder_chunk, &c) != 0) {
        free(c.data);
        return NULL;
    }
    *out_frames = c.frames;
    return c.data;
}
