/*
 * nemotron_asr_decoder.c - RNN-T greedy decoder.
 * 2-layer LSTM prediction network with hidden/cell state preserved across
 * chunks, the joint network (encoder + prediction projections, ReLU), and
 * greedy argmax over the vocabulary classifier, emitting SentencePiece text.
 */
#include "nemotron_asr.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * Consecutive all-blank encoder frames that mark the decoder as "stalled" and
 * trigger a code-switch recovery reset (see nemo_rnnt_stream_accept). One frame
 * is 80 ms, so 16 frames is ~1.28 s: past any brief inter-word gap, short
 * enough to recover the onset of a switched-to language. A deliberate pause
 * this long does re-prime the decoder (harmless - it just starts fresh); raise
 * the threshold via NEMO_STALL_FRAMES if that is undesirable for your input.
 */
#define NEMO_STALL_RESET_FRAMES 16

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} strbuf_t;

struct nemo_rnnt_stream_t {
    float *h;
    float *c;
    float *enc_proj;
    int enc_proj_cap;
    float pred_out[NEMO_PRED_HIDDEN];
    float pred_proj[NEMO_JOINT_HIDDEN];
    strbuf_t text;
    int blank_run;      /* consecutive all-blank frames (code-switch stall detector) */
    int emitted_any;    /* have we emitted any token yet (gates the reset) */
    int stall_frames;   /* per-stream stall threshold, resolved once at create */
    int enc_reset_req;  /* set when a stall reset fires; the caller clears it and
                         * forwards it to the encoder via
                         * nemo_encoder_stream_request_reset (see the orchestration
                         * in nemotron_asr.c). Kept on the stream, not the shared
                         * ctx, so concurrent streams never cross-signal. */
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

    /* SentencePiece's leading-space marker "▁" is the 3-byte UTF-8 E2 96 81.
     * Compare byte-by-byte rather than memcmp(...,3): the && short-circuits at
     * the NUL of a shorter piece, so a 1- or 2-byte token is never read past
     * its allocation. */
    const unsigned char *up = (const unsigned char *)piece;
    if (up[0] == 0xE2 && up[1] == 0x96 && up[2] == 0x81) {
        const char *rest = piece + 3;
        if (sb->len > 0 && sb->data[sb->len - 1] != ' ') {
            if (sb_append_bytes(sb, " ", 1) != 0) return -1;
        }
        return sb_append_bytes(sb, rest, strlen(rest));
    }
    return sb_append_bytes(sb, piece, strlen(piece));
}

static void lstm_layer_step(const float *input, const nemo_weight_t *w_ih, const nemo_weight_t *w_hh,
                            const float *b_ih, const float *b_hh,
                            float *h, float *c) {
    float gates[4 * NEMO_PRED_HIDDEN];
    nemo_lstm_gates_weight(gates, input, h, w_ih, w_hh, b_ih, b_hh,
                           NEMO_PRED_HIDDEN, 4 * NEMO_PRED_HIDDEN);
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
    lstm_layer_step(in0, &d->lstm_w_ih[0], &d->lstm_w_hh[0], d->lstm_b_ih[0], d->lstm_b_hh[0],
                    h, c);
    memcpy(tmp0, h, sizeof(tmp0));
    lstm_layer_step(tmp0, &d->lstm_w_ih[1], &d->lstm_w_hh[1], d->lstm_b_ih[1], d->lstm_b_hh[1],
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
    return nemo_argmax_matvec_weight(hidden, &j->out_w, j->out_b,
                                     NEMO_JOINT_HIDDEN, NEMO_VOCAB_WITH_BLANK, NULL);
}

void nemo_rnnt_stream_free(nemo_rnnt_stream_t *s) {
    if (!s) return;
    free(s->h);
    free(s->c);
    free(s->enc_proj);
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
    nemo_linear_weight(s->pred_proj, s->pred_out, &j->pred_w, j->pred_b,
                       1, NEMO_PRED_HIDDEN, NEMO_JOINT_HIDDEN);
    if (sb_reserve(&s->text, 256) != 0) {
        nemo_rnnt_stream_free(s);
        return NULL;
    }
    s->text.data[0] = 0;
    /* Resolve the stall threshold once per stream (not a process-wide latch):
     * NEMO_STALL_RESET_FRAMES by default, overridable per run via the
     * NEMO_STALL_FRAMES env var (0 disables code-switch recovery). */
    s->stall_frames = NEMO_STALL_RESET_FRAMES;
    {
        const char *e = getenv("NEMO_STALL_FRAMES");
        if (e && e[0]) {
            int v = atoi(e);
            s->stall_frames = v > 0 ? v : 0;
        }
    }
    return s;
}

/*
 * Consume the pending "reset the encoder caches" request raised by a stall
 * reset. The caller (the streaming orchestration) forwards it to the encoder
 * via nemo_encoder_stream_request_reset. Returns 1 once per stall reset.
 */
int nemo_rnnt_stream_take_enc_reset(nemo_rnnt_stream_t *s) {
    if (!s) return 0;
    int r = s->enc_reset_req;
    s->enc_reset_req = 0;
    return r;
}

int nemo_rnnt_stream_accept(nemo_ctx_t *ctx, nemo_rnnt_stream_t *s,
                            const float *enc, int enc_frames) {
    const nemo_joint_t *j = &ctx->model.joint;
    if (enc_frames <= 0) return 0;
    if (enc_frames > s->enc_proj_cap) {
        int nc = s->enc_proj_cap ? s->enc_proj_cap * 2 : 8;
        while (nc < enc_frames) nc *= 2;
        float *p = (float *)realloc(s->enc_proj, (size_t)nc * NEMO_JOINT_HIDDEN * sizeof(float));
        if (!p) return -1;
        s->enc_proj = p;
        s->enc_proj_cap = nc;
    }
    float *enc_proj = s->enc_proj;
    nemo_linear_weight(enc_proj, enc, &j->enc_w, j->enc_b, enc_frames, NEMO_D_MODEL, NEMO_JOINT_HIDDEN);
    for (int t = 0; t < enc_frames; t++) {
        int emitted = 0;
        /*
         * Code-switch recovery. After a language switch the decoder stalls: the
         * prediction network still "speaks" the previous language and the
         * encoder's attention cache still holds its context, so the new
         * language's onset decodes as a run of blanks and gets dropped. We take
         * a blank run of stall_frames (~1.28 s) as "the model lost the thread"
         * and re-prime the prediction network to its blank/start state, and
         * flag the encoder to clear its caches, so decoding re-locks onto the
         * current audio.
         *
         * The condition uses `%` on purpose: through a long stall it re-fires
         * every stall_frames frames, which the switched-to onset needs (a single
         * shot recovers the near onset but not a longer, encoder-side stall).
         * It is gated on emitted_any so leading silence never clips the first
         * word. Genuine silence is safe (a re-primed decoder still emits blanks
         * for it). The tradeoff: a deliberate >=stall_frames same-language pause
         * also re-primes, i.e. the next word is decoded with fresh conditioning
         * - normally harmless (like starting a new sentence), occasionally a
         * casing/subword difference. Raise NEMO_STALL_FRAMES to relax it.
         */
        if (s->stall_frames > 0 && s->emitted_any &&
            s->blank_run > 0 && s->blank_run % s->stall_frames == 0) {
            memset(s->h, 0, (size_t)NEMO_PRED_LAYERS * NEMO_PRED_HIDDEN * sizeof(float));
            memset(s->c, 0, (size_t)NEMO_PRED_LAYERS * NEMO_PRED_HIDDEN * sizeof(float));
            pred_step(ctx, -1, s->h, s->c, s->pred_out);
            nemo_linear_weight(s->pred_proj, s->pred_out, &j->pred_w, j->pred_b,
                               1, NEMO_PRED_HIDDEN, NEMO_JOINT_HIDDEN);
            s->enc_reset_req = 1;
        }
        while (emitted < ctx->max_symbols_per_step) {
            int tok = joint_argmax(ctx, enc_proj + (size_t)t * NEMO_JOINT_HIDDEN, s->pred_proj);
            if (tok == NEMO_BLANK_ID) break;
            if (append_piece(ctx, &s->text, tok) != 0) return -1;
            ctx->perf_tokens++;
            s->emitted_any = 1;
            if (pred_step(ctx, tok, s->h, s->c, s->pred_out) != 0) return -1;
            nemo_linear_weight(s->pred_proj, s->pred_out, &j->pred_w, j->pred_b,
                               1, NEMO_PRED_HIDDEN, NEMO_JOINT_HIDDEN);
            emitted++;
        }
        s->blank_run = emitted ? 0 : s->blank_run + 1;
    }
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
