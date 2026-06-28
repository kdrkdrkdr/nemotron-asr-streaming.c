/*
 * nemotron_asr.c - top-level transcription orchestration.
 * Language-prompt table, context load/free, and the chunked streaming loop that
 * feeds audio through the mel -> encoder -> RNN-T graph for both WAV-file and
 * live stdin input.
 */
#include "nemotron_asr.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

typedef struct {
    const char *name;
    int id;
} lang_entry_t;

static const lang_entry_t LANGS[] = {
    {"en-US",0},{"en",0},{"en-GB",1},{"enGB",1},{"es-ES",2},{"esES",2},{"es-US",3},{"es",3},
    {"zh-CN",4},{"zh-ZH",4},{"zh-TW",5},{"hi-IN",6},{"hi",6},{"hi-HI",6},{"ar-AR",7},{"ar",7},
    {"fr-FR",8},{"fr",8},{"de-DE",9},{"de",9},{"ja-JP",10},{"ja-JA",10},{"ru-RU",11},{"ru",11},
    {"pt-BR",12},{"pt-PT",13},{"pt",13},{"ko-KR",14},{"ko",14},{"ko-KO",14},{"it-IT",15},{"it",15},
    {"nl-NL",16},{"nl",16},{"pl-PL",17},{"pl",17},{"tr-TR",18},{"tr",18},{"uk-UA",19},{"uk",19},
    {"ro-RO",20},{"ro",20},{"el-GR",21},{"el",21},{"cs-CZ",22},{"cs",22},{"hu-HU",23},{"hu",23},
    {"sv-SE",24},{"sv",24},{"da-DK",25},{"da",25},{"fi-FI",26},{"fi",26},{"no-NO",27},{"no",27},
    {"nb-NO",103},{"nb",103},{"nn-NO",104},{"nn",104},{"sk-SK",28},{"sk",28},{"hr-HR",29},{"hr",29},
    {"bg-BG",30},{"bg",30},{"lt-LT",31},{"lt",31},{"et-EE",60},{"et",60},{"lv-LV",61},{"lv",61},
    {"sl-SI",62},{"sl",62},{"th-TH",32},{"vi-VN",33},{"id-ID",34},{"ms-MY",35},{"bn-IN",36},
    {"ur-PK",37},{"fa-IR",38},{"ta-IN",39},{"te-IN",40},{"mr-IN",41},{"gu-IN",42},{"kn-IN",43},
    {"ml-IN",44},{"si-LK",45},{"ne-NP",46},{"km-KH",47},{"sw-KE",48},{"am-ET",49},{"ha-NG",50},
    {"zu-ZA",51},{"yo-NG",52},{"ig-NG",53},{"af-ZA",54},{"rw-RW",55},{"so-SO",56},{"ny-MW",57},
    {"ln-CD",58},{"or-KE",59},{"he-IL",64},{"ku-TR",65},{"az-AZ",66},{"ka-GE",67},{"hy-AM",68},
    {"uz-UZ",69},{"tg-TJ",70},{"ky-KG",71},{"qu-PE",80},{"ay-BO",81},{"gn-PY",82},{"nah-MX",83},
    {"mi-NZ",96},{"haw-US",97},{"sm-WS",98},{"to-TO",99},{"fr-CA",100},{"mt-MT",102},{"auto",101},
};

nemo_ctx_t *nemo_load(const char *model_path) {
    nemo_ctx_t *ctx = (nemo_ctx_t *)calloc(1, sizeof(nemo_ctx_t));
    if (!ctx) return NULL;
    if (nemo_model_load(&ctx->model, model_path) != 0) {
        free(ctx);
        return NULL;
    }
    snprintf(ctx->model_path, sizeof(ctx->model_path), "%s", model_path);
    ctx->prompt_id = 101;
    ctx->att_left = 56;
    ctx->att_right = 3;
    ctx->strip_lang_tags = 0;
    ctx->max_symbols_per_step = 10;
    fprintf(stderr, "Loaded Nemotron 3.5 ASR model: %d tensors, vocab=%d\n",
            ctx->model.n_tensors, ctx->model.vocab_size);
    return ctx;
}

void nemo_free(nemo_ctx_t *ctx) {
    if (!ctx) return;
    nemo_model_free(&ctx->model);
    free(ctx);
}

int nemo_set_language(nemo_ctx_t *ctx, const char *lang) {
    if (!ctx || !lang) return -1;
    for (size_t i = 0; i < sizeof(LANGS) / sizeof(LANGS[0]); i++) {
        if (strcmp(LANGS[i].name, lang) == 0) {
            ctx->prompt_id = LANGS[i].id;
            return 0;
        }
    }
    fprintf(stderr, "nemotron: unknown language '%s' (try en-US, ko-KR, auto)\n", lang);
    return -1;
}

typedef struct {
    nemo_ctx_t *ctx;
    nemo_rnnt_stream_t *rnnt;
    nemo_encoder_stream_t *enc;
} transcribe_stream_cb_t;

static int transcribe_encoder_chunk(void *user, const float *enc, int enc_frames) {
    transcribe_stream_cb_t *s = (transcribe_stream_cb_t *)user;
    double t0 = nemo_time_ms();
    int rc = nemo_rnnt_stream_accept(s->ctx, s->rnnt, enc, enc_frames);
    s->ctx->perf_decoder_ms += nemo_time_ms() - t0;
    if (rc == 0) s->ctx->perf_frames += enc_frames;
    return rc;
}

static int transcribe_mel_chunk(void *user, const float *mel, int mel_frames, int final) {
    transcribe_stream_cb_t *s = (transcribe_stream_cb_t *)user;
    double dec_before = s->ctx->perf_decoder_ms;
    double t0 = nemo_time_ms();
    int rc = nemo_encoder_stream_accept(s->ctx, s->enc, mel, mel_frames, final,
                                        transcribe_encoder_chunk, s);
    double elapsed = nemo_time_ms() - t0;
    s->ctx->perf_encoder_ms += elapsed - (s->ctx->perf_decoder_ms - dec_before);
    return rc;
}

char *nemo_transcribe_audio(nemo_ctx_t *ctx, const float *samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0) return NULL;
    ctx->perf_audio_ms = 1000.0 * (double)n_samples / (double)NEMO_SAMPLE_RATE;
    ctx->perf_tokens = 0;
    ctx->perf_frames = 0;
    ctx->perf_mel_ms = 0.0;
    ctx->perf_encoder_ms = 0.0;
    ctx->perf_decoder_ms = 0.0;

    nemo_rnnt_stream_t *rnnt = nemo_rnnt_stream_create(ctx);
    nemo_encoder_stream_t *enc = nemo_encoder_stream_create(ctx);
    nemo_mel_stream_t *mel_stream = nemo_mel_stream_create(ctx);
    if (!rnnt || !enc || !mel_stream) {
        nemo_rnnt_stream_free(rnnt);
        nemo_encoder_stream_free(enc);
        nemo_mel_stream_free(mel_stream);
        return NULL;
    }
    transcribe_stream_cb_t cb = {ctx, rnnt, enc};
    int sample_step = NEMO_HOP_LENGTH * NEMO_SUBSAMPLING_FACTOR * (ctx->att_right + 1);
    if (sample_step < NEMO_HOP_LENGTH) sample_step = NEMO_HOP_LENGTH;
    int rc = 0;
    for (int off = 0; off < n_samples; off += sample_step) {
        int n = n_samples - off;
        if (n > sample_step) n = sample_step;
        int final = (off + n == n_samples);
        double enc_before = ctx->perf_encoder_ms;
        double dec_before = ctx->perf_decoder_ms;
        double t0 = nemo_time_ms();
        rc = nemo_mel_stream_accept(mel_stream, samples + off, n, final, transcribe_mel_chunk, &cb);
        double elapsed = nemo_time_ms() - t0;
        ctx->perf_mel_ms += elapsed - (ctx->perf_encoder_ms - enc_before) - (ctx->perf_decoder_ms - dec_before);
        if (rc != 0) break;
    }
    nemo_mel_stream_free(mel_stream);
    nemo_encoder_stream_free(enc);
    if (rc != 0) { nemo_rnnt_stream_free(rnnt); return NULL; }
    char *text = nemo_rnnt_stream_finish(rnnt);
    return text;
}

/*
 * Live streaming from stdin: raw little-endian s16, 16 kHz, mono. Feeds the
 * same cache-aware streaming graph as nemo_transcribe_audio, but reads
 * incrementally and writes recognized text deltas to stdout as they are
 * emitted. Returns the full final text (caller frees).
 */
char *nemo_transcribe_stdin(nemo_ctx_t *ctx) {
    if (!ctx) return NULL;
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY); /* stop CRLF translation mangling s16le */
#endif
    ctx->perf_audio_ms = 0.0;
    ctx->perf_tokens = 0;
    ctx->perf_frames = 0;
    ctx->perf_mel_ms = 0.0;
    ctx->perf_encoder_ms = 0.0;
    ctx->perf_decoder_ms = 0.0;

    nemo_rnnt_stream_t *rnnt = nemo_rnnt_stream_create(ctx);
    nemo_encoder_stream_t *enc = nemo_encoder_stream_create(ctx);
    nemo_mel_stream_t *mel_stream = nemo_mel_stream_create(ctx);
    int sample_step = NEMO_HOP_LENGTH * NEMO_SUBSAMPLING_FACTOR * (ctx->att_right + 1);
    if (sample_step < NEMO_HOP_LENGTH) sample_step = NEMO_HOP_LENGTH;
    int16_t *pcm = (int16_t *)malloc((size_t)sample_step * sizeof(int16_t));
    float *fbuf = (float *)malloc((size_t)sample_step * sizeof(float));
    if (!rnnt || !enc || !mel_stream || !pcm || !fbuf) {
        nemo_rnnt_stream_free(rnnt);
        nemo_encoder_stream_free(enc);
        nemo_mel_stream_free(mel_stream);
        free(pcm);
        free(fbuf);
        return NULL;
    }

    transcribe_stream_cb_t cb = {ctx, rnnt, enc};
    size_t printed = 0;
    long long total_samples = 0;
    int rc = 0;
    int read_err = 0;
    for (;;) {
        size_t got = fread(pcm, sizeof(int16_t), (size_t)sample_step, stdin);
        read_err = ferror(stdin);
        int final = feof(stdin) || read_err;
        for (size_t i = 0; i < got; i++) fbuf[i] = (float)pcm[i] / 32768.0f;
        total_samples += (long long)got;
        rc = nemo_mel_stream_accept(mel_stream, fbuf, (int)got, final, transcribe_mel_chunk, &cb);
        size_t len = nemo_rnnt_stream_text_len(rnnt);
        if (len > printed) {
            const char *t = nemo_rnnt_stream_text(rnnt);
            fwrite(t + printed, 1, len - printed, stdout);
            fflush(stdout);
            printed = len;
        }
        if (rc != 0 || final) break;
    }
    if (read_err) fprintf(stderr, "nemotron: stdin read error\n");

    ctx->perf_audio_ms = 1000.0 * (double)total_samples / (double)NEMO_SAMPLE_RATE;
    nemo_mel_stream_free(mel_stream);
    nemo_encoder_stream_free(enc);
    free(pcm);
    free(fbuf);
    if (rc != 0 || read_err) {
        nemo_rnnt_stream_free(rnnt);
        return NULL;
    }
    return nemo_rnnt_stream_finish(rnnt);
}

char *nemo_transcribe(nemo_ctx_t *ctx, const char *wav_path) {
    int n = 0;
    float *samples = nemo_load_wav(wav_path, &n);
    if (!samples) return NULL;
    char *text = nemo_transcribe_audio(ctx, samples, n);
    free(samples);
    return text;
}
