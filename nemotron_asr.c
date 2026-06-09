#include "nemotron_asr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

char *nemo_transcribe_audio(nemo_ctx_t *ctx, const float *samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0) return NULL;
    ctx->perf_audio_ms = 1000.0 * (double)n_samples / (double)NEMO_SAMPLE_RATE;
    ctx->perf_tokens = 0;

    double t0 = nemo_time_ms();
    int mel_frames = 0;
    float *mel = nemo_mel_spectrogram(ctx, samples, n_samples, &mel_frames);
    ctx->perf_mel_ms = nemo_time_ms() - t0;
    if (!mel) return NULL;

    t0 = nemo_time_ms();
    int enc_frames = 0;
    float *enc = nemo_encoder_forward(ctx, mel, mel_frames, &enc_frames);
    ctx->perf_encoder_ms = nemo_time_ms() - t0;
    free(mel);
    if (!enc) return NULL;
    ctx->perf_frames = enc_frames;

    t0 = nemo_time_ms();
    char *text = nemo_rnnt_greedy_decode(ctx, enc, enc_frames);
    ctx->perf_decoder_ms = nemo_time_ms() - t0;
    free(enc);
    return text;
}

char *nemo_transcribe(nemo_ctx_t *ctx, const char *wav_path) {
    int n = 0;
    float *samples = nemo_load_wav(wav_path, &n);
    if (!samples) return NULL;
    char *text = nemo_transcribe_audio(ctx, samples, n);
    free(samples);
    return text;
}
