/*
 * nemotron_asr_audio.c - audio front end.
 * WAV loading (PCM s16, mono downmix, Kaiser windowed-sinc resampling to
 * 16 kHz) and the streaming log-mel spectrogram (n_fft=512, win=400, hop=160,
 * 128 mel bins).
 */
#include "nemotron_asr.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct nemo_mel_stream_t {
    const nemo_ctx_t *ctx;
    float *samples;
    int n_samples;
    int cap_samples;
    int next_frame;
};

static uint16_t le16(const unsigned char *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t le32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* Modified Bessel function I0, for the Kaiser resampling window. */
static double nemo_bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    for (int k = 1; k < 32; k++) {
        double t = x / (2.0 * (double)k);
        term *= t * t;
        sum += term;
        if (term < 1e-12 * sum) break;
    }
    return sum;
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
    if (out_frames < 1) out_frames = 1;
    float *out = nemo_alloc((size_t)out_frames, sizeof(float));
    if (!out) { free(mono); return NULL; }
    /*
     * Windowed-sinc (Kaiser) resampling. The sinc cutoff is lowered to the
     * destination Nyquist when downsampling, which anti-aliases (linear
     * interpolation does not). Each output sample is normalized by the sum of
     * its tap weights, which also corrects for taps truncated at the edges.
     */
    const int half = 16; /* 32-tap filter */
    const double beta = 6.0;
    const double ratio = (double)NEMO_SAMPLE_RATE / (double)sample_rate;
    const double cutoff = ratio < 1.0 ? ratio : 1.0;
    const double inv_i0 = 1.0 / nemo_bessel_i0(beta);
    for (int i = 0; i < out_frames; i++) {
        double center = (double)i / ratio; /* output i maps to this input position */
        int c = (int)floor(center);
        double frac = center - (double)c;
        double acc = 0.0, wsum = 0.0;
        for (int k = -half + 1; k <= half; k++) {
            int idx = c + k;
            if (idx < 0 || idx >= in_frames) continue;
            double x = (double)k - frac; /* distance in input samples from output point */
            double sx = M_PI * cutoff * x;
            double sinc = (x == 0.0) ? 1.0 : sin(sx) / sx;
            double wn = x / (double)half; /* Kaiser window argument in (-1, 1) */
            double win = (wn > -1.0 && wn < 1.0)
                             ? nemo_bessel_i0(beta * sqrt(1.0 - wn * wn)) * inv_i0
                             : 0.0;
            double w = sinc * win;
            acc += w * (double)mono[idx];
            wsum += w;
        }
        out[i] = (float)(wsum != 0.0 ? acc / wsum : 0.0);
    }
    free(mono);
    *out_n_samples = out_frames;
    return out;
}

static int mel_available_frames(int n_samples, int final) {
    if (final) {
        int frames = n_samples / NEMO_HOP_LENGTH + 1;
        return frames < 1 ? 1 : frames;
    }
    int last = n_samples - 1 - (NEMO_WIN_LENGTH / 2 - 1);
    if (last < 0) return 0;
    return last / NEMO_HOP_LENGTH + 1;
}

static void compute_mel_frame(const nemo_ctx_t *ctx, const float *samples, int n_samples,
                              int frame_idx, float *out) {
    const nemo_encoder_t *e = &ctx->model.encoder;
    float power[NEMO_N_FFT / 2 + 1];
    float frame[NEMO_N_FFT];
    int start = frame_idx * NEMO_HOP_LENGTH - NEMO_N_FFT / 2;
    int win_offset = (NEMO_N_FFT - NEMO_WIN_LENGTH) / 2;
    memset(frame, 0, sizeof(frame));
    for (int i = 0; i < NEMO_WIN_LENGTH; i++) {
        int src = start + win_offset + i;
        float s = (src >= 0 && src < n_samples) ? samples[src] : 0.0f;
        frame[win_offset + i] = s * e->window[i];
    }
    nemo_fft512_power_f32(power, frame);
    for (int m = 0; m < NEMO_MEL_BINS; m++) {
        const float *fb = e->mel_fb + (size_t)m * (NEMO_N_FFT / 2 + 1);
        float v = nemo_dot_f32(fb, power, NEMO_N_FFT / 2 + 1);
        out[m] = logf((float)v + 5.960464477539063e-08f);
    }
}

nemo_mel_stream_t *nemo_mel_stream_create(const nemo_ctx_t *ctx) {
    nemo_mel_stream_t *s = (nemo_mel_stream_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->ctx = ctx;
    return s;
}

int nemo_mel_stream_accept(nemo_mel_stream_t *s, const float *samples, int n_samples,
                           int final, nemo_mel_chunk_cb cb, void *user) {
    if (!s || !cb || n_samples < 0) return -1;
    if (n_samples > 0) {
        if (s->n_samples + n_samples > s->cap_samples) {
            int nc = s->cap_samples ? s->cap_samples * 2 : 8192;
            while (nc < s->n_samples + n_samples) nc *= 2;
            float *p = (float *)realloc(s->samples, (size_t)nc * sizeof(float));
            if (!p) return -1;
            s->samples = p;
            s->cap_samples = nc;
        }
        memcpy(s->samples + s->n_samples, samples, (size_t)n_samples * sizeof(float));
        s->n_samples += n_samples;
    }

    int avail = mel_available_frames(s->n_samples, final);
    int new_frames = avail - s->next_frame;
    if (new_frames > 0) {
        float *mel = nemo_alloc((size_t)new_frames * NEMO_MEL_BINS, sizeof(float));
        if (!mel) return -1;
        for (int t = 0; t < new_frames; t++) {
            float tmp[NEMO_MEL_BINS];
            compute_mel_frame(s->ctx, s->samples, s->n_samples, s->next_frame + t, tmp);
            for (int m = 0; m < NEMO_MEL_BINS; m++) {
                mel[(size_t)m * new_frames + t] = tmp[m];
            }
        }
        s->next_frame = avail;
        int rc = cb(user, mel, new_frames, final);
        free(mel);
        if (rc != 0) return -1;
    } else if (final) {
        if (cb(user, NULL, 0, final) != 0) return -1;
    }
    return 0;
}

void nemo_mel_stream_free(nemo_mel_stream_t *s) {
    if (!s) return;
    free(s->samples);
    free(s);
}

float *nemo_mel_spectrogram(const nemo_ctx_t *ctx, const float *samples, int n_samples, int *out_frames) {
    const nemo_encoder_t *e = &ctx->model.encoder;
    int frames = n_samples / NEMO_HOP_LENGTH + 1;
    if (frames < 1) frames = 1;
    float *mel = nemo_alloc((size_t)frames * NEMO_MEL_BINS, sizeof(float));
    if (!mel) return NULL;

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
        nemo_fft512_power_f32(power, frame);
        for (int m = 0; m < NEMO_MEL_BINS; m++) {
            const float *fb = e->mel_fb + (size_t)m * (NEMO_N_FFT / 2 + 1);
            float v = nemo_dot_f32(fb, power, NEMO_N_FFT / 2 + 1);
            mel[(size_t)m * frames + t] = logf((float)v + 5.960464477539063e-08f);
        }
    }
    *out_frames = frames;
    return mel;
}
