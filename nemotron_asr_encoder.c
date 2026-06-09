#include "nemotron_asr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LN_EPS 1.0e-5f
#define NEG_INF -10000.0f

static int conv_out_len(int n, int left, int right, int k, int stride) {
    return (n + left + right - k) / stride + 1;
}

static void conv2d(float *out, const float *in, const float *w, const float *b,
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
    (void)right;
}

static void conv1d_depthwise_causal(float *out, const float *in, const float *w,
                                    int t, int dim, int k, int left) {
    for (int tt = 0; tt < t; tt++) {
        for (int c = 0; c < dim; c++) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; kk++) {
                int src = tt + kk - left;
                if (src >= 0 && src < t) sum += in[(size_t)src * dim + c] * w[(size_t)c * k + kk];
            }
            out[(size_t)tt * dim + c] = sum;
        }
    }
}

static int pre_encode(const nemo_ctx_t *ctx, const float *mel, int frames, float **out_x, int *out_t) {
    const nemo_encoder_t *e = &ctx->model.encoder;
    int t0 = frames, f0 = NEMO_MEL_BINS;
    float *x0 = nemo_alloc((size_t)t0 * f0, sizeof(float));
    if (!x0) return -1;
    for (int t = 0; t < t0; t++) {
        for (int f = 0; f < f0; f++) x0[(size_t)t * f0 + f] = mel[(size_t)f * frames + t];
    }

    int t1 = conv_out_len(t0, 2, 1, 3, 2);
    int f1 = conv_out_len(f0, 2, 1, 3, 2);
    float *x1 = nemo_alloc((size_t)NEMO_CONV_CHANNELS * t1 * f1, sizeof(float));
    if (!x1) { free(x0); return -1; }
    conv2d(x1, x0, e->pre_conv0_w, e->pre_conv0_b, 1, NEMO_CONV_CHANNELS, t0, f0, 3, 2, 2, 1, 1);
    nemo_relu(x1, NEMO_CONV_CHANNELS * t1 * f1);
    free(x0);

    int t2 = conv_out_len(t1, 2, 1, 3, 2);
    int f2 = conv_out_len(f1, 2, 1, 3, 2);
    float *x2 = nemo_alloc((size_t)NEMO_CONV_CHANNELS * t2 * f2, sizeof(float));
    float *x3 = nemo_alloc((size_t)NEMO_CONV_CHANNELS * t2 * f2, sizeof(float));
    if (!x2 || !x3) { free(x1); free(x2); free(x3); return -1; }
    conv2d(x2, x1, e->pre_conv2_w, e->pre_conv2_b,
           NEMO_CONV_CHANNELS, NEMO_CONV_CHANNELS, t1, f1, 3, 2, 2, 1, NEMO_CONV_CHANNELS);
    free(x1);
    conv2d(x3, x2, e->pre_conv3_w, e->pre_conv3_b,
           NEMO_CONV_CHANNELS, NEMO_CONV_CHANNELS, t2, f2, 1, 1, 0, 0, 1);
    free(x2);
    nemo_relu(x3, NEMO_CONV_CHANNELS * t2 * f2);

    int t4 = conv_out_len(t2, 2, 1, 3, 2);
    int f4 = conv_out_len(f2, 2, 1, 3, 2);
    float *x4 = nemo_alloc((size_t)NEMO_CONV_CHANNELS * t4 * f4, sizeof(float));
    float *x5 = nemo_alloc((size_t)NEMO_CONV_CHANNELS * t4 * f4, sizeof(float));
    if (!x4 || !x5) { free(x3); free(x4); free(x5); return -1; }
    conv2d(x4, x3, e->pre_conv5_w, e->pre_conv5_b,
           NEMO_CONV_CHANNELS, NEMO_CONV_CHANNELS, t2, f2, 3, 2, 2, 1, NEMO_CONV_CHANNELS);
    free(x3);
    conv2d(x5, x4, e->pre_conv6_w, e->pre_conv6_b,
           NEMO_CONV_CHANNELS, NEMO_CONV_CHANNELS, t4, f4, 1, 1, 0, 0, 1);
    free(x4);
    nemo_relu(x5, NEMO_CONV_CHANNELS * t4 * f4);

    if (f4 != NEMO_SUB_FREQ) {
        fprintf(stderr, "nemotron: unexpected subsampling frequency %d (expected %d)\n", f4, NEMO_SUB_FREQ);
        free(x5);
        return -1;
    }

    float *flat = nemo_alloc((size_t)t4 * NEMO_PREENC_IN, sizeof(float));
    float *proj = nemo_alloc((size_t)t4 * NEMO_D_MODEL, sizeof(float));
    if (!flat || !proj) { free(x5); free(flat); free(proj); return -1; }
    for (int t = 0; t < t4; t++) {
        for (int c = 0; c < NEMO_CONV_CHANNELS; c++) {
            for (int f = 0; f < f4; f++) {
                flat[(size_t)t * NEMO_PREENC_IN + c * f4 + f] =
                    x5[((size_t)c * t4 + t) * f4 + f];
            }
        }
    }
    free(x5);
    nemo_linear(proj, flat, e->pre_out_w, e->pre_out_b, t4, NEMO_PREENC_IN, NEMO_D_MODEL);
    free(flat);
    *out_x = proj;
    *out_t = t4;
    return 0;
}

static void make_rel_pe(float *pe, int t) {
    int pos_len = 2 * t - 1;
    for (int r = 0; r < pos_len; r++) {
        float pos = (float)(t - 1 - r);
        for (int i = 0; i < NEMO_D_MODEL / 2; i++) {
            float div = expf(-(logf(10000.0f) / (float)NEMO_D_MODEL) * (float)(2 * i));
            pe[(size_t)r * NEMO_D_MODEL + 2 * i] = sinf(pos * div);
            pe[(size_t)r * NEMO_D_MODEL + 2 * i + 1] = cosf(pos * div);
        }
    }
}

static int att_allowed(const nemo_ctx_t *ctx, int q, int k) {
    int chunk = ctx->att_right + 1;
    int left_chunks = ctx->att_left >= 0 ? ctx->att_left / chunk : 1000000;
    int qc = q / chunk;
    int kc = k / chunk;
    int diff = qc - kc;
    return diff >= 0 && diff <= left_chunks;
}

static int rel_attention(const nemo_ctx_t *ctx, const nemo_enc_layer_t *l,
                         const float *x, const float *pe, int t, float *out) {
    float *q = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    float *k = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    float *v = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    float *p = nemo_alloc((size_t)(2 * t - 1) * NEMO_D_MODEL, sizeof(float));
    float *ctxv = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    float *scores = nemo_alloc((size_t)t, sizeof(float));
    int *idx = (int *)malloc((size_t)t * sizeof(int));
    if (!q || !k || !v || !p || !ctxv || !scores || !idx) {
        free(q); free(k); free(v); free(p); free(ctxv); free(scores); free(idx);
        return -1;
    }
    nemo_linear_nobias(q, x, l->att_q_w, t, NEMO_D_MODEL, NEMO_D_MODEL);
    nemo_linear_nobias(k, x, l->att_k_w, t, NEMO_D_MODEL, NEMO_D_MODEL);
    nemo_linear_nobias(v, x, l->att_v_w, t, NEMO_D_MODEL, NEMO_D_MODEL);
    nemo_linear_nobias(p, pe, l->att_pos_w, 2 * t - 1, NEMO_D_MODEL, NEMO_D_MODEL);

    const float scale = 1.0f / sqrtf((float)NEMO_HEAD_DIM);
    for (int qi = 0; qi < t; qi++) {
        for (int h = 0; h < NEMO_ENC_HEADS; h++) {
            int n = 0;
            for (int kj = 0; kj < t; kj++) {
                if (!att_allowed(ctx, qi, kj)) continue;
                const float *qh = q + (size_t)qi * NEMO_D_MODEL + h * NEMO_HEAD_DIM;
                const float *kh = k + (size_t)kj * NEMO_D_MODEL + h * NEMO_HEAD_DIM;
                int pidx = (t - 1) - qi + kj;
                const float *ph = p + (size_t)pidx * NEMO_D_MODEL + h * NEMO_HEAD_DIM;
                const float *bu = l->pos_bias_u + h * NEMO_HEAD_DIM;
                const float *bv = l->pos_bias_v + h * NEMO_HEAD_DIM;
                float ac = 0.0f, bd = 0.0f;
                for (int d = 0; d < NEMO_HEAD_DIM; d++) {
                    ac += (qh[d] + bu[d]) * kh[d];
                    bd += (qh[d] + bv[d]) * ph[d];
                }
                scores[n] = (ac + bd) * scale;
                idx[n] = kj;
                n++;
            }
            if (n == 0) {
                for (int d = 0; d < NEMO_HEAD_DIM; d++) ctxv[(size_t)qi * NEMO_D_MODEL + h * NEMO_HEAD_DIM + d] = 0.0f;
                continue;
            }
            nemo_softmax(scores, n);
            for (int d = 0; d < NEMO_HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int a = 0; a < n; a++) {
                    const float *vh = v + (size_t)idx[a] * NEMO_D_MODEL + h * NEMO_HEAD_DIM;
                    sum += scores[a] * vh[d];
                }
                ctxv[(size_t)qi * NEMO_D_MODEL + h * NEMO_HEAD_DIM + d] = sum;
            }
        }
    }
    nemo_linear_nobias(out, ctxv, l->att_out_w, t, NEMO_D_MODEL, NEMO_D_MODEL);
    free(q); free(k); free(v); free(p); free(ctxv); free(scores); free(idx);
    return 0;
}

static int conformer_conv(const nemo_enc_layer_t *l, const float *x, int t, float *out) {
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
    conv1d_depthwise_causal(dw, glu, l->conv_dw_w, t, NEMO_D_MODEL, 9, 8);
    nemo_layer_norm(norm, dw, l->conv_norm_w, l->conv_norm_b, t, NEMO_D_MODEL, LN_EPS);
    nemo_swish(norm, t * NEMO_D_MODEL);
    nemo_linear_nobias(out, norm, l->conv_pw2_w, t, NEMO_D_MODEL, NEMO_D_MODEL);
    free(pw); free(glu); free(dw); free(norm);
    return 0;
}

static int conformer_layer(nemo_ctx_t *ctx, const nemo_enc_layer_t *l,
                           float *x, const float *pe, int t) {
    float *norm = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    float *ff = nemo_alloc((size_t)t * NEMO_FFN_DIM, sizeof(float));
    float *tmp = nemo_alloc((size_t)t * NEMO_D_MODEL, sizeof(float));
    if (!norm || !ff || !tmp) { free(norm); free(ff); free(tmp); return -1; }

    nemo_layer_norm(norm, x, l->norm_ff1_w, l->norm_ff1_b, t, NEMO_D_MODEL, LN_EPS);
    nemo_linear_nobias(ff, norm, l->ff1_linear1_w, t, NEMO_D_MODEL, NEMO_FFN_DIM);
    nemo_swish(ff, t * NEMO_FFN_DIM);
    nemo_linear_nobias(tmp, ff, l->ff1_linear2_w, t, NEMO_FFN_DIM, NEMO_D_MODEL);
    for (int i = 0; i < t * NEMO_D_MODEL; i++) x[i] += 0.5f * tmp[i];

    nemo_layer_norm(norm, x, l->norm_att_w, l->norm_att_b, t, NEMO_D_MODEL, LN_EPS);
    if (rel_attention(ctx, l, norm, pe, t, tmp) != 0) { free(norm); free(ff); free(tmp); return -1; }
    for (int i = 0; i < t * NEMO_D_MODEL; i++) x[i] += tmp[i];

    nemo_layer_norm(norm, x, l->norm_conv_w, l->norm_conv_b, t, NEMO_D_MODEL, LN_EPS);
    if (conformer_conv(l, norm, t, tmp) != 0) { free(norm); free(ff); free(tmp); return -1; }
    for (int i = 0; i < t * NEMO_D_MODEL; i++) x[i] += tmp[i];

    nemo_layer_norm(norm, x, l->norm_ff2_w, l->norm_ff2_b, t, NEMO_D_MODEL, LN_EPS);
    nemo_linear_nobias(ff, norm, l->ff2_linear1_w, t, NEMO_D_MODEL, NEMO_FFN_DIM);
    nemo_swish(ff, t * NEMO_FFN_DIM);
    nemo_linear_nobias(tmp, ff, l->ff2_linear2_w, t, NEMO_FFN_DIM, NEMO_D_MODEL);
    for (int i = 0; i < t * NEMO_D_MODEL; i++) x[i] += 0.5f * tmp[i];

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
            for (int i = 0; i < NEMO_D_MODEL; i++) sum += xr[i] * wr[i];
            hr[o] = sum > 0.0f ? sum : 0.0f;
        }
    }
    nemo_linear(y, h, e->prompt2_w, e->prompt2_b, t, 2 * NEMO_D_MODEL, NEMO_D_MODEL);
    memcpy(x, y, (size_t)t * NEMO_D_MODEL * sizeof(float));
    free(h); free(y);
    return 0;
}

float *nemo_encoder_forward(nemo_ctx_t *ctx, const float *mel, int mel_frames, int *out_frames) {
    float *x = NULL;
    int t = 0;
    if (pre_encode(ctx, mel, mel_frames, &x, &t) != 0) return NULL;
    float *pe = nemo_alloc((size_t)(2 * t - 1) * NEMO_D_MODEL, sizeof(float));
    if (!pe) { free(x); return NULL; }
    make_rel_pe(pe, t);
    for (int i = 0; i < NEMO_ENC_LAYERS; i++) {
        if (conformer_layer(ctx, &ctx->model.encoder.layers[i], x, pe, t) != 0) {
            free(pe);
            free(x);
            return NULL;
        }
    }
    free(pe);
    if (apply_prompt(ctx, x, t) != 0) {
        free(x);
        return NULL;
    }
    *out_frames = t;
    return x;
}
