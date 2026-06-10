#include "nemotron_asr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} strbuf_t;

struct nemo_rnnt_stream_t {
    float *h;
    float *c;
    float pred_out[NEMO_PRED_HIDDEN];
    float pred_proj[NEMO_JOINT_HIDDEN];
    strbuf_t text;
};

static int sb_reserve(strbuf_t *sb, size_t add) {
    if (sb->len + add + 1 <= sb->cap) return 0;
    size_t nc = sb->cap ? sb->cap * 2 : 256;
    while (nc < sb->len + add + 1) nc *= 2;
    char *p = (char *)realloc(sb->data, nc);
    if (!p) return -1;
    sb->data = p;
    sb->cap = nc;
    return 0;
}

static int sb_append_bytes(strbuf_t *sb, const char *s, size_t n) {
    if (n == 0) return 0;
    if (sb_reserve(sb, n) != 0) return -1;
    memcpy(sb->data + sb->len, s, n);
    sb->len += n;
    sb->data[sb->len] = 0;
    return 0;
}

static int is_lang_tag(const char *s) {
    size_t n = strlen(s);
    if (n < 4 || s[0] != '<' || s[n - 1] != '>') return 0;
    for (size_t i = 1; i + 1 < n; i++) {
        char c = s[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '-')) return 0;
    }
    return 1;
}

static int append_piece(nemo_ctx_t *ctx, strbuf_t *sb, int token) {
    if (token < 0 || token >= ctx->model.vocab_size) return 0;
    const char *piece = ctx->model.vocab[token];
    if (!piece || strcmp(piece, "<unk>") == 0) return 0;
    if (ctx->strip_lang_tags && is_lang_tag(piece)) return 0;

    const unsigned char spm_space[] = {0xE2, 0x96, 0x81, 0};
    if (memcmp(piece, spm_space, 3) == 0) {
        const char *rest = piece + 3;
        if (sb->len > 0 && sb->data[sb->len - 1] != ' ') {
            if (sb_append_bytes(sb, " ", 1) != 0) return -1;
        }
        return sb_append_bytes(sb, rest, strlen(rest));
    }
    return sb_append_bytes(sb, piece, strlen(piece));
}

static void lstm_layer_step(const float *input, const float *w_ih, const float *w_hh,
                            const float *b_ih, const float *b_hh,
                            float *h, float *c) {
    float gates[4 * NEMO_PRED_HIDDEN];
    float recurrent[4 * NEMO_PRED_HIDDEN];
    nemo_matvec_f32(gates, input, w_ih, b_ih, NEMO_PRED_HIDDEN, 4 * NEMO_PRED_HIDDEN);
    nemo_matvec_f32(recurrent, h, w_hh, b_hh, NEMO_PRED_HIDDEN, 4 * NEMO_PRED_HIDDEN);
    nemo_vec_axpy_inplace(gates, recurrent, 1.0f, 4 * NEMO_PRED_HIDDEN);
    for (int i = 0; i < NEMO_PRED_HIDDEN; i++) {
        float in_gate = nemo_sigmoid(gates[i]);
        float forget_gate = nemo_sigmoid(gates[NEMO_PRED_HIDDEN + i]);
        float cell_gate = tanhf(gates[2 * NEMO_PRED_HIDDEN + i]);
        float out_gate = nemo_sigmoid(gates[3 * NEMO_PRED_HIDDEN + i]);
        c[i] = forget_gate * c[i] + in_gate * cell_gate;
        h[i] = out_gate * tanhf(c[i]);
    }
}

static int pred_step(const nemo_ctx_t *ctx, int token, float *h, float *c, float *out) {
    const nemo_decoder_t *d = &ctx->model.decoder;
    float in0[NEMO_PRED_HIDDEN];
    float tmp0[NEMO_PRED_HIDDEN];
    if (token >= 0) {
        if (token >= NEMO_VOCAB_WITH_BLANK) return -1;
        memcpy(in0, d->embed_w + (size_t)token * NEMO_PRED_HIDDEN, sizeof(in0));
    } else {
        memset(in0, 0, sizeof(in0));
    }
    lstm_layer_step(in0, d->lstm_w_ih[0], d->lstm_w_hh[0], d->lstm_b_ih[0], d->lstm_b_hh[0],
                    h, c);
    memcpy(tmp0, h, sizeof(tmp0));
    lstm_layer_step(tmp0, d->lstm_w_ih[1], d->lstm_w_hh[1], d->lstm_b_ih[1], d->lstm_b_hh[1],
                    h + NEMO_PRED_HIDDEN, c + NEMO_PRED_HIDDEN);
    memcpy(out, h + NEMO_PRED_HIDDEN, sizeof(float) * NEMO_PRED_HIDDEN);
    return 0;
}

static int joint_argmax(const nemo_ctx_t *ctx, const float *enc_proj, const float *pred_proj) {
    const nemo_joint_t *j = &ctx->model.joint;
    float hidden[NEMO_JOINT_HIDDEN];
    for (int i = 0; i < NEMO_JOINT_HIDDEN; i++) {
        float v = enc_proj[i] + pred_proj[i];
        hidden[i] = v > 0.0f ? v : 0.0f;
    }
    return nemo_argmax_matvec_f32(hidden, j->out_w, j->out_b,
                                  NEMO_JOINT_HIDDEN, NEMO_VOCAB_WITH_BLANK, NULL);
}

void nemo_rnnt_stream_free(nemo_rnnt_stream_t *s) {
    if (!s) return;
    free(s->h);
    free(s->c);
    free(s->text.data);
    free(s);
}

nemo_rnnt_stream_t *nemo_rnnt_stream_create(nemo_ctx_t *ctx) {
    const nemo_joint_t *j = &ctx->model.joint;
    nemo_rnnt_stream_t *s = (nemo_rnnt_stream_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->h = nemo_alloc((size_t)NEMO_PRED_LAYERS * NEMO_PRED_HIDDEN, sizeof(float));
    s->c = nemo_alloc((size_t)NEMO_PRED_LAYERS * NEMO_PRED_HIDDEN, sizeof(float));
    if (!s->h || !s->c) {
        nemo_rnnt_stream_free(s);
        return NULL;
    }
    if (pred_step(ctx, -1, s->h, s->c, s->pred_out) != 0) {
        nemo_rnnt_stream_free(s);
        return NULL;
    }
    nemo_linear(s->pred_proj, s->pred_out, j->pred_w, j->pred_b,
                1, NEMO_PRED_HIDDEN, NEMO_JOINT_HIDDEN);
    if (sb_reserve(&s->text, 256) != 0) {
        nemo_rnnt_stream_free(s);
        return NULL;
    }
    s->text.data[0] = 0;
    return s;
}

int nemo_rnnt_stream_accept(nemo_ctx_t *ctx, nemo_rnnt_stream_t *s,
                            const float *enc, int enc_frames) {
    const nemo_joint_t *j = &ctx->model.joint;
    float *enc_proj = nemo_alloc((size_t)enc_frames * NEMO_JOINT_HIDDEN, sizeof(float));
    if (!enc_proj) return -1;
    nemo_linear(enc_proj, enc, j->enc_w, j->enc_b, enc_frames, NEMO_D_MODEL, NEMO_JOINT_HIDDEN);
    for (int t = 0; t < enc_frames; t++) {
        int emitted = 0;
        while (emitted < ctx->max_symbols_per_step) {
            int tok = joint_argmax(ctx, enc_proj + (size_t)t * NEMO_JOINT_HIDDEN, s->pred_proj);
            if (tok == NEMO_BLANK_ID) break;
            if (append_piece(ctx, &s->text, tok) != 0) { free(enc_proj); return -1; }
            ctx->perf_tokens++;
            if (pred_step(ctx, tok, s->h, s->c, s->pred_out) != 0) { free(enc_proj); return -1; }
            nemo_linear(s->pred_proj, s->pred_out, j->pred_w, j->pred_b,
                        1, NEMO_PRED_HIDDEN, NEMO_JOINT_HIDDEN);
            emitted++;
        }
    }
    free(enc_proj);
    return 0;
}

const char *nemo_rnnt_stream_text(const nemo_rnnt_stream_t *s) {
    return (s && s->text.data) ? s->text.data : "";
}

size_t nemo_rnnt_stream_text_len(const nemo_rnnt_stream_t *s) {
    return (s && s->text.data) ? s->text.len : 0;
}

char *nemo_rnnt_stream_finish(nemo_rnnt_stream_t *s) {
    char *text = s->text.data;
    s->text.data = NULL;
    nemo_rnnt_stream_free(s);
    if (!text) text = (char *)calloc(1, 1);
    return text;
}

char *nemo_rnnt_greedy_decode(nemo_ctx_t *ctx, const float *enc, int enc_frames) {
    nemo_rnnt_stream_t *stream = nemo_rnnt_stream_create(ctx);
    if (!stream) return NULL;
    if (nemo_rnnt_stream_accept(ctx, stream, enc, enc_frames) != 0) {
        nemo_rnnt_stream_free(stream);
        return NULL;
    }
    return nemo_rnnt_stream_finish(stream);
}
