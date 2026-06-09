#include "nemotron_asr.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static uint16_t le16(const unsigned char *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t le32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

float *nemo_load_wav(const char *path, int *out_n_samples) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "nemotron: cannot open wav %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 44) {
        fclose(f);
        return NULL;
    }
    unsigned char *buf = (unsigned char *)malloc((size_t)sz);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    if (memcmp(buf, "RIFF", 4) != 0 || memcmp(buf + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "nemotron: %s is not RIFF/WAVE\n", path);
        free(buf);
        return NULL;
    }

    int channels = 0, sample_rate = 0, bits = 0, format = 0;
    const unsigned char *pcm = NULL;
    uint32_t pcm_bytes = 0;
    size_t off = 12;
    while (off + 8 <= (size_t)sz) {
        const unsigned char *ch = buf + off;
        uint32_t csz = le32(ch + 4);
        off += 8;
        if (off + csz > (size_t)sz) break;
        if (memcmp(ch, "fmt ", 4) == 0 && csz >= 16) {
            format = le16(buf + off);
            channels = le16(buf + off + 2);
            sample_rate = (int)le32(buf + off + 4);
            bits = le16(buf + off + 14);
        } else if (memcmp(ch, "data", 4) == 0) {
            pcm = buf + off;
            pcm_bytes = csz;
        }
        off += csz + (csz & 1u);
    }
    if (format != 1 || bits != 16 || channels < 1 || !pcm) {
        fprintf(stderr, "nemotron: unsupported wav format=%d bits=%d channels=%d\n", format, bits, channels);
        free(buf);
        return NULL;
    }
    int in_frames = (int)(pcm_bytes / (uint32_t)(channels * 2));
    float *mono = nemo_alloc((size_t)in_frames, sizeof(float));
    if (!mono) { free(buf); return NULL; }
    for (int i = 0; i < in_frames; i++) {
        int sum = 0;
        for (int c = 0; c < channels; c++) {
            const unsigned char *s = pcm + (size_t)(i * channels + c) * 2;
            sum += (int)(int16_t)le16(s);
        }
        mono[i] = (float)sum / (32768.0f * (float)channels);
    }
    free(buf);

    if (sample_rate == NEMO_SAMPLE_RATE) {
        *out_n_samples = in_frames;
        return mono;
    }

    int out_frames = (int)llround((double)in_frames * (double)NEMO_SAMPLE_RATE / (double)sample_rate);
    float *out = nemo_alloc((size_t)out_frames, sizeof(float));
    if (!out) { free(mono); return NULL; }
    for (int i = 0; i < out_frames; i++) {
        double src = (double)i * (double)sample_rate / (double)NEMO_SAMPLE_RATE;
        int j = (int)floor(src);
        double a = src - (double)j;
        float s0 = (j >= 0 && j < in_frames) ? mono[j] : 0.0f;
        float s1 = (j + 1 >= 0 && j + 1 < in_frames) ? mono[j + 1] : s0;
        out[i] = (float)((1.0 - a) * s0 + a * s1);
    }
    free(mono);
    *out_n_samples = out_frames;
    return out;
}

float *nemo_mel_spectrogram(const nemo_ctx_t *ctx, const float *samples, int n_samples, int *out_frames) {
    const nemo_encoder_t *e = &ctx->model.encoder;
    int frames = n_samples / NEMO_HOP_LENGTH + 1;
    if (frames < 1) frames = 1;
    float *mel = nemo_alloc((size_t)frames * NEMO_MEL_BINS, sizeof(float));
    if (!mel) return NULL;

    float re[NEMO_N_FFT / 2 + 1];
    float im[NEMO_N_FFT / 2 + 1];
    float power[NEMO_N_FFT / 2 + 1];
    float frame[NEMO_N_FFT];

    for (int t = 0; t < frames; t++) {
        int start = t * NEMO_HOP_LENGTH - NEMO_N_FFT / 2;
        memset(frame, 0, sizeof(frame));
        int win_offset = (NEMO_N_FFT - NEMO_WIN_LENGTH) / 2;
        for (int i = 0; i < NEMO_WIN_LENGTH; i++) {
            int src = start + win_offset + i;
            float s = (src >= 0 && src < n_samples) ? samples[src] : 0.0f;
            frame[win_offset + i] = s * e->window[i];
        }
        for (int k = 0; k <= NEMO_N_FFT / 2; k++) {
            double r = 0.0, ii = 0.0;
            for (int n = 0; n < NEMO_N_FFT; n++) {
                double ang = -2.0 * M_PI * (double)k * (double)n / (double)NEMO_N_FFT;
                r += (double)frame[n] * cos(ang);
                ii += (double)frame[n] * sin(ang);
            }
            re[k] = (float)r;
            im[k] = (float)ii;
            power[k] = re[k] * re[k] + im[k] * im[k];
        }
        for (int m = 0; m < NEMO_MEL_BINS; m++) {
            const float *fb = e->mel_fb + (size_t)m * (NEMO_N_FFT / 2 + 1);
            double v = 0.0;
            for (int k = 0; k <= NEMO_N_FFT / 2; k++) v += (double)fb[k] * power[k];
            mel[(size_t)m * frames + t] = logf((float)v + 5.960464477539063e-08f);
        }
    }
    *out_frames = frames;
    return mel;
}
