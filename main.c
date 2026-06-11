#include "nemotron_asr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr, "Nemotron 3.5 ASR Streaming pure C runtime\n\n");
    fprintf(stderr, "Usage: %s -m model.bin -i audio.wav [options]\n\n", argv0);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m <file>       Converted Nemotron model bin\n");
    fprintf(stderr, "  -i <file>       Input WAV (PCM s16; resampled to 16 kHz if needed)\n");
    fprintf(stderr, "  -t <n>          Number of worker threads (default: all CPUs, max 16)\n");
    fprintf(stderr, "  -l <lang>       Language prompt, e.g. en-US, ko-KR, auto (default auto)\n");
    fprintf(stderr, "  --att-right N   Right context in 80 ms encoder frames: 0,1,3,6,13 (default 3)\n");
    fprintf(stderr, "  --strip-tags    Remove emitted language tags like <en-US>\n");
    fprintf(stderr, "  --model-info    Load model and print tensor/vocab summary only\n");
    fprintf(stderr, "  --max-symbols N Max non-blank RNNT labels per frame (default 10)\n");
}

int main(int argc, char **argv) {
    const char *model = NULL;
    const char *input = NULL;
    const char *lang = "auto";
    int model_info = 0;
    int strip_tags = 0;
    int att_right = 3;
    int max_symbols = 10;
    int n_threads = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            lang = argv[++i];
        } else if (strcmp(argv[i], "--att-right") == 0 && i + 1 < argc) {
            att_right = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-symbols") == 0 && i + 1 < argc) {
            max_symbols = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--strip-tags") == 0) {
            strip_tags = 1;
        } else if (strcmp(argv[i], "--model-info") == 0) {
            model_info = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (!model || (!input && !model_info)) {
        usage(argv[0]);
        return 1;
    }
    if (n_threads < 0) {
        fprintf(stderr, "nemotron: -t must be >= 1\n");
        return 1;
    }
    if (n_threads == 0) n_threads = nemo_get_num_cpus();
    nemo_set_threads(n_threads);

    nemo_ctx_t *ctx = nemo_load(model);
    if (!ctx) return 1;
    if (nemo_set_language(ctx, lang) != 0) {
        nemo_free(ctx);
        return 1;
    }
    if (att_right != 0 && att_right != 1 && att_right != 3 && att_right != 6 && att_right != 13) {
        fprintf(stderr, "nemotron: --att-right must be one of 0,1,3,6,13\n");
        nemo_free(ctx);
        return 1;
    }
    ctx->att_right = att_right;
    ctx->att_left = 56;
    ctx->strip_lang_tags = strip_tags;
    if (max_symbols > 0) ctx->max_symbols_per_step = max_symbols;

    if (model_info) {
        int f32_tensors = 0;
        int bf16_tensors = 0;
        int q8_tensors = 0;
        uint64_t f32_bytes = 0;
        uint64_t bf16_bytes = 0;
        uint64_t q8_bytes = 0;
        for (int i = 0; i < ctx->model.n_tensors; i++) {
            if (ctx->model.tensors[i].dtype == NEMO_TENSOR_BF16) {
                bf16_tensors++;
                bf16_bytes += ctx->model.tensors[i].nbytes;
            } else if (ctx->model.tensors[i].dtype == NEMO_TENSOR_Q8) {
                q8_tensors++;
                q8_bytes += ctx->model.tensors[i].nbytes;
            } else {
                f32_tensors++;
                f32_bytes += ctx->model.tensors[i].nbytes;
            }
        }
        printf("Nemotron 3.5 ASR model\n");
        printf("  tensors: %d\n", ctx->model.n_tensors);
        printf("  f32:     %d tensors, %.2f GiB\n", f32_tensors, (double)f32_bytes / (1024.0 * 1024.0 * 1024.0));
        printf("  bf16:    %d tensors, %.2f GiB\n", bf16_tensors, (double)bf16_bytes / (1024.0 * 1024.0 * 1024.0));
        printf("  q8:      %d tensors, %.2f GiB\n", q8_tensors, (double)q8_bytes / (1024.0 * 1024.0 * 1024.0));
        printf("  vocab:   %d (+ blank id %d)\n", ctx->model.vocab_size, NEMO_BLANK_ID);
        printf("  encoder: FastConformer cache-aware, layers=%d, d_model=%d, heads=%d\n",
               NEMO_ENC_LAYERS, NEMO_D_MODEL, NEMO_ENC_HEADS);
        printf("  decoder: RNNT LSTM layers=%d hidden=%d\n", NEMO_PRED_LAYERS, NEMO_PRED_HIDDEN);
        nemo_free(ctx);
        return 0;
    }

    double t0 = nemo_time_ms();
    char *text = nemo_transcribe(ctx, input);
    double total = nemo_time_ms() - t0;
    if (!text) {
        nemo_free(ctx);
        return 1;
    }
    printf("%s\n", text);
    fprintf(stderr,
            "Inference: %.0f ms, frames=%d, tokens=%d (mel %.0f ms, encoder %.0f ms, decoder %.0f ms)\n",
            total, ctx->perf_frames, ctx->perf_tokens,
            ctx->perf_mel_ms, ctx->perf_encoder_ms, ctx->perf_decoder_ms);
    if (ctx->perf_audio_ms > 0.0 && total > 0.0) {
        fprintf(stderr, "Audio: %.2f s processed in %.2f s (%.2fx realtime)\n",
                ctx->perf_audio_ms / 1000.0, total / 1000.0, ctx->perf_audio_ms / total);
    }
    free(text);
    nemo_free(ctx);
    return 0;
}
