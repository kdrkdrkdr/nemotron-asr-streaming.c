/*
 * main.c - CLI entry point for the Nemotron 3.5 ASR runtime.
 * Parses options, loads the model, and transcribes either a WAV file (-i) or a
 * live raw-s16le 16 kHz mono stdin stream (--stdin).
 */
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
    fprintf(stderr, "  --stdin         Stream raw s16le 16 kHz mono audio from stdin (live input)\n");
    fprintf(stderr, "  -t <n>          Number of worker threads (default 1, max 16)\n");
    fprintf(stderr, "  -l <lang>       Language prompt, e.g. en-US, ko-KR, auto (default auto)\n");
    fprintf(stderr, "  --att-right N   Right context in 80 ms encoder frames: 0,1,3,6 (default 3)\n");
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
    int use_stdin = 0;

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
        } else if (strcmp(argv[i], "--stdin") == 0) {
            use_stdin = 1;
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

    if (!model || (!input && !model_info && !use_stdin)) {
        usage(argv[0]);
        return 1;
    }
    if (input && use_stdin) {
        fprintf(stderr, "nemotron: use either -i <file> or --stdin, not both\n");
        return 1;
    }
    if (n_threads < 0) {
        fprintf(stderr, "nemotron: -t must be >= 1\n");
        return 1;
    }
    if (n_threads == 0) n_threads = 1;
    nemo_set_threads(n_threads);

    nemo_ctx_t *ctx = nemo_load(model);
    if (!ctx) return 1;
    if (nemo_set_language(ctx, lang) != 0) {
        nemo_free(ctx);
        return 1;
    }
    if (!model_info &&
        att_right != 0 && att_right != 1 && att_right != 3 && att_right != 6) {
        /*
         * The model was also trained with a 14-frame (att_right 13, 1120 ms)
         * context family, but that chunk is too coarse for the code-switch
         * recovery reset (which lands on a chunk boundary), so a switched-to
         * language can still lose its onset there. We only expose the low-latency
         * families that recover cleanly; att_right 6 already gives 560 ms of
         * look-ahead for accuracy-first use.
         */
        fprintf(stderr, "nemotron: --att-right must be one of 0,1,3,6\n");
        nemo_free(ctx);
        return 1;
    }
    ctx->att_right = att_right;
    ctx->att_left = 56;
    ctx->strip_lang_tags = strip_tags;
    if (max_symbols > 0) ctx->max_symbols_per_step = max_symbols;

    if (model_info) {
        int f32_tensors = 0;
        int q8p_tensors = 0;
        uint64_t f32_bytes = 0;
        uint64_t q8p_bytes = 0;
        for (int i = 0; i < ctx->model.n_tensors; i++) {
            if (ctx->model.tensors[i].dtype == NEMO_TENSOR_Q8P) {
                q8p_tensors++;
                q8p_bytes += ctx->model.tensors[i].nbytes;
            } else {
                f32_tensors++;
                f32_bytes += ctx->model.tensors[i].nbytes;
            }
        }
        printf("Nemotron 3.5 ASR model\n");
        printf("  tensors: %d\n", ctx->model.n_tensors);
        printf("  f32:     %d tensors, %.2f GiB\n", f32_tensors, (double)f32_bytes / (1024.0 * 1024.0 * 1024.0));
        printf("  q8p:     %d tensors, %.2f GiB\n",
               q8p_tensors, (double)q8p_bytes / (1024.0 * 1024.0 * 1024.0));
        printf("  vocab:   %d (+ blank id %d)\n", ctx->model.vocab_size, NEMO_BLANK_ID);
        printf("  encoder: FastConformer cache-aware, layers=%d, d_model=%d, heads=%d\n",
               NEMO_ENC_LAYERS, NEMO_D_MODEL, NEMO_ENC_HEADS);
        printf("  decoder: RNNT LSTM layers=%d hidden=%d\n", NEMO_PRED_LAYERS, NEMO_PRED_HIDDEN);
        nemo_free(ctx);
        return 0;
    }

    double t0 = nemo_time_ms();
    char *text = use_stdin ? nemo_transcribe_stdin(ctx) : nemo_transcribe(ctx, input);
    double total = nemo_time_ms() - t0;
    if (!text) {
        nemo_free(ctx);
        return 1;
    }
    if (use_stdin) {
        /* deltas were already streamed to stdout; just end the line */
        printf("\n");
    } else {
        printf("%s\n", text);
    }
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
