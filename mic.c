#include "nemotron_asr.h"

#if defined(__APPLE__)

#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>
#include <ctype.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MIC_NUM_BUFFERS 3

static volatile sig_atomic_t g_stop = 0;

typedef struct {
    float *data;
    size_t len;
    size_t cap;
    size_t dropped;
    size_t pushed;
    float peak_since_read;
    pthread_mutex_t lock;
} sample_fifo_t;

typedef struct {
    nemo_ctx_t *ctx;
    nemo_mel_stream_t *mel;
    nemo_encoder_stream_t *enc;
    nemo_rnnt_stream_t *rnnt;
    size_t printed_len;
    int trace;
    int samples_seen;
    int mel_chunks;
    int encoder_chunks;
    int enc_frames_total;
} live_asr_t;

typedef struct {
    sample_fifo_t *fifo;
    AudioQueueRef queue;
    AudioQueueBufferRef buffers[MIC_NUM_BUFFERS];
    int stopping;
} mic_capture_t;

typedef struct {
    AudioDeviceID id;
    char name[256];
    char uid[256];
    int channels;
    int is_default;
} input_device_t;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

static void sleep_ms(int ms) {
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

static int parse_int(const char *s, int *out) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || (end && *end)) return -1;
    if (v < -2147483647L || v > 2147483647L) return -1;
    *out = (int)v;
    return 0;
}

static int valid_att_right(int n) {
    return n == 0 || n == 1 || n == 3 || n == 6 || n == 13;
}

static int cfstring_to_cstr(CFStringRef s, char *dst, size_t cap) {
    if (!dst || cap == 0) return -1;
    dst[0] = 0;
    if (!s) return -1;
    if (CFStringGetCString(s, dst, cap, kCFStringEncodingUTF8)) return 0;
    return -1;
}

static int get_device_string(AudioDeviceID id, AudioObjectPropertySelector selector,
                             char *dst, size_t cap) {
    CFStringRef value = NULL;
    UInt32 size = sizeof(value);
    AudioObjectPropertyAddress addr = {
        selector,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    OSStatus err = AudioObjectGetPropertyData(id, &addr, 0, NULL, &size, &value);
    if (err != noErr || !value) {
        if (dst && cap > 0) dst[0] = 0;
        return -1;
    }
    int rc = cfstring_to_cstr(value, dst, cap);
    CFRelease(value);
    return rc;
}

static int input_channel_count(AudioDeviceID id) {
    AudioObjectPropertyAddress addr = {
        kAudioDevicePropertyStreamConfiguration,
        kAudioDevicePropertyScopeInput,
        kAudioObjectPropertyElementMain
    };
    UInt32 size = 0;
    OSStatus err = AudioObjectGetPropertyDataSize(id, &addr, 0, NULL, &size);
    if (err != noErr || size == 0) return 0;

    AudioBufferList *abl = (AudioBufferList *)malloc(size);
    if (!abl) return 0;
    err = AudioObjectGetPropertyData(id, &addr, 0, NULL, &size, abl);
    if (err != noErr) {
        free(abl);
        return 0;
    }

    int channels = 0;
    for (UInt32 i = 0; i < abl->mNumberBuffers; i++) {
        channels += (int)abl->mBuffers[i].mNumberChannels;
    }
    free(abl);
    return channels;
}

static AudioDeviceID default_input_device(void) {
    AudioDeviceID id = kAudioObjectUnknown;
    UInt32 size = sizeof(id);
    AudioObjectPropertyAddress addr = {
        kAudioHardwarePropertyDefaultInputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &size, &id) != noErr) {
        return kAudioObjectUnknown;
    }
    return id;
}

static int collect_input_devices(input_device_t **out_devices, int *out_count) {
    *out_devices = NULL;
    *out_count = 0;

    AudioObjectPropertyAddress addr = {
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    UInt32 size = 0;
    OSStatus err = AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &addr, 0, NULL, &size);
    if (err != noErr || size == 0) return -1;

    int n_ids = (int)(size / sizeof(AudioDeviceID));
    AudioDeviceID *ids = (AudioDeviceID *)malloc(size);
    if (!ids) return -1;
    err = AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &size, ids);
    if (err != noErr) {
        free(ids);
        return -1;
    }

    AudioDeviceID default_id = default_input_device();
    input_device_t *devices = NULL;
    int n_devices = 0;
    int cap_devices = 0;
    for (int i = 0; i < n_ids; i++) {
        int channels = input_channel_count(ids[i]);
        if (channels <= 0) continue;

        if (n_devices == cap_devices) {
            int nc = cap_devices ? cap_devices * 2 : 8;
            input_device_t *tmp = (input_device_t *)realloc(devices, (size_t)nc * sizeof(*devices));
            if (!tmp) {
                free(devices);
                free(ids);
                return -1;
            }
            devices = tmp;
            cap_devices = nc;
        }

        input_device_t *d = &devices[n_devices++];
        memset(d, 0, sizeof(*d));
        d->id = ids[i];
        d->channels = channels;
        d->is_default = (ids[i] == default_id);
        if (get_device_string(ids[i], kAudioObjectPropertyName, d->name, sizeof(d->name)) != 0) {
            snprintf(d->name, sizeof(d->name), "AudioDevice-%u", (unsigned)ids[i]);
        }
        if (get_device_string(ids[i], kAudioDevicePropertyDeviceUID, d->uid, sizeof(d->uid)) != 0) {
            d->uid[0] = 0;
        }
    }

    free(ids);
    *out_devices = devices;
    *out_count = n_devices;
    return 0;
}

static void print_input_devices(void) {
    input_device_t *devices = NULL;
    int n = 0;
    if (collect_input_devices(&devices, &n) != 0) {
        fprintf(stderr, "nemotron_mic: cannot list input devices\n");
        return;
    }
    printf("Input devices:\n");
    for (int i = 0; i < n; i++) {
        printf("  [%d] id=%u channels=%d%s\n", i, (unsigned)devices[i].id,
               devices[i].channels, devices[i].is_default ? " default" : "");
        printf("      name: %s\n", devices[i].name);
        printf("      uid:  %s\n", devices[i].uid[0] ? devices[i].uid : "(none)");
    }
    free(devices);
}

static int arg_is_integer(const char *s) {
    if (!s || !*s) return 0;
    if (*s == '-' || *s == '+') s++;
    if (!*s) return 0;
    while (*s) {
        if (!isdigit((unsigned char)*s)) return 0;
        s++;
    }
    return 1;
}

static int resolve_input_device(const char *arg, CFStringRef *out_uid,
                                char *desc, size_t desc_cap) {
    *out_uid = NULL;
    if (desc && desc_cap > 0) desc[0] = 0;
    if (!arg || !*arg) return 0;

    input_device_t *devices = NULL;
    int n = 0;
    if (collect_input_devices(&devices, &n) != 0) {
        fprintf(stderr, "nemotron_mic: cannot inspect input devices\n");
        return -1;
    }

    const input_device_t *chosen = NULL;
    if (arg_is_integer(arg)) {
        char *end = NULL;
        long v = strtol(arg, &end, 10);
        if (end && *end == 0) {
            if (v >= 0 && v < n) {
                chosen = &devices[v];
            } else {
                for (int i = 0; i < n; i++) {
                    if ((long)devices[i].id == v) {
                        chosen = &devices[i];
                        break;
                    }
                }
            }
        }
    }

    if (!chosen) {
        for (int i = 0; i < n; i++) {
            if ((devices[i].uid[0] && strcmp(devices[i].uid, arg) == 0) ||
                strcmp(devices[i].name, arg) == 0) {
                chosen = &devices[i];
                break;
            }
        }
    }

    if (!chosen || !chosen->uid[0]) {
        fprintf(stderr, "nemotron_mic: input device not found: %s\n", arg);
        fprintf(stderr, "Run `./nemotron_asr_mic --list-devices` and pass an index, id, uid, or exact name.\n");
        free(devices);
        return -1;
    }

    if (desc && desc_cap > 0) {
        snprintf(desc, desc_cap, "[%ld] id=%u %s",
                 (long)(chosen - devices), (unsigned)chosen->id, chosen->name);
    }
    *out_uid = CFStringCreateWithCString(NULL, chosen->uid, kCFStringEncodingUTF8);
    free(devices);
    return *out_uid ? 0 : -1;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s -m model.bin [options]\n\n"
            "Options:\n"
            "  -m <file>          Converted Nemotron model bin\n"
            "  --list-devices     List macOS input devices and exit\n"
            "  --device X         Input device index, AudioDeviceID, UID, or exact name\n"
            "  -l <lang>          Language prompt, e.g. en-US, ko-KR, auto (default auto)\n"
            "  --att-right N      Right context in 80 ms encoder frames: 0,1,3,6,13 (default 3)\n"
            "  --push-frames N    Microphone push size in 80 ms encoder-frame units (default att_right+1)\n"
            "  --seconds N        Stop automatically after N seconds (default until Ctrl-C)\n"
            "  --strip-tags       Remove emitted language tags like <en-US>\n"
            "  --max-symbols N    Max non-blank RNNT labels per frame (default 10)\n"
            "  --buffer-ms N      AudioQueue buffer size in milliseconds (default 80)\n"
            "  --queue-sec N      Max queued microphone audio before dropping oldest samples (default 20)\n"
            "  --meter            Print captured-audio queue and peak level every second\n"
            "  --no-trace         Print transcript only, without chunk trace lines\n",
            argv0);
}

static int fifo_init(sample_fifo_t *f, size_t cap) {
    memset(f, 0, sizeof(*f));
    f->data = (float *)malloc(cap * sizeof(float));
    if (!f->data) return -1;
    f->cap = cap;
    if (pthread_mutex_init(&f->lock, NULL) != 0) {
        free(f->data);
        memset(f, 0, sizeof(*f));
        return -1;
    }
    return 0;
}

static void fifo_free(sample_fifo_t *f) {
    if (!f) return;
    pthread_mutex_destroy(&f->lock);
    free(f->data);
    memset(f, 0, sizeof(*f));
}

static void fifo_push_i16(sample_fifo_t *f, const int16_t *pcm, size_t n) {
    if (n == 0 || f->cap == 0) return;
    float peak = 0.0f;
    pthread_mutex_lock(&f->lock);
    if (n >= f->cap) {
        size_t start = n - f->cap;
        f->dropped += f->len + start;
        for (size_t i = 0; i < f->cap; i++) {
            float v = (float)pcm[start + i] / 32768.0f;
            float a = v < 0.0f ? -v : v;
            if (a > peak) peak = a;
            f->data[i] = v;
        }
        f->len = f->cap;
        f->pushed += n;
        if (peak > f->peak_since_read) f->peak_since_read = peak;
        pthread_mutex_unlock(&f->lock);
        return;
    }
    if (f->len + n > f->cap) {
        size_t drop = f->len + n - f->cap;
        memmove(f->data, f->data + drop, (f->len - drop) * sizeof(float));
        f->len -= drop;
        f->dropped += drop;
    }
    for (size_t i = 0; i < n; i++) {
        float v = (float)pcm[i] / 32768.0f;
        float a = v < 0.0f ? -v : v;
        if (a > peak) peak = a;
        f->data[f->len + i] = v;
    }
    f->len += n;
    f->pushed += n;
    if (peak > f->peak_since_read) f->peak_since_read = peak;
    pthread_mutex_unlock(&f->lock);
}

static int fifo_pop_exact(sample_fifo_t *f, float *out, size_t n) {
    int ok = 0;
    pthread_mutex_lock(&f->lock);
    if (f->len >= n) {
        memcpy(out, f->data, n * sizeof(float));
        f->len -= n;
        if (f->len > 0) memmove(f->data, f->data + n, f->len * sizeof(float));
        ok = 1;
    }
    pthread_mutex_unlock(&f->lock);
    return ok;
}

static size_t fifo_pop_all(sample_fifo_t *f, float **out) {
    size_t n = 0;
    *out = NULL;
    pthread_mutex_lock(&f->lock);
    if (f->len > 0) {
        float *p = (float *)malloc(f->len * sizeof(float));
        if (p) {
            n = f->len;
            memcpy(p, f->data, n * sizeof(float));
            f->len = 0;
            *out = p;
        }
    }
    pthread_mutex_unlock(&f->lock);
    return n;
}

static size_t fifo_dropped(sample_fifo_t *f) {
    size_t n;
    pthread_mutex_lock(&f->lock);
    n = f->dropped;
    pthread_mutex_unlock(&f->lock);
    return n;
}

static void fifo_stats_take(sample_fifo_t *f, size_t *queued, size_t *pushed, float *peak) {
    pthread_mutex_lock(&f->lock);
    if (queued) *queued = f->len;
    if (pushed) *pushed = f->pushed;
    if (peak) *peak = f->peak_since_read;
    f->peak_since_read = 0.0f;
    pthread_mutex_unlock(&f->lock);
}

static void print_text_delta(live_asr_t *s) {
    const char *text = nemo_rnnt_stream_text(s->rnnt);
    size_t len = nemo_rnnt_stream_text_len(s->rnnt);
    if (len > s->printed_len) {
        fwrite(text + s->printed_len, 1, len - s->printed_len, stdout);
        fflush(stdout);
        s->printed_len = len;
    }
}

static int live_encoder_chunk(void *user, const float *enc, int enc_frames) {
    live_asr_t *s = (live_asr_t *)user;
    int tokens_before = s->ctx->perf_tokens;
    double t0 = nemo_time_ms();
    int rc = nemo_rnnt_stream_accept(s->ctx, s->rnnt, enc, enc_frames);
    s->ctx->perf_decoder_ms += nemo_time_ms() - t0;
    if (rc != 0) return rc;

    s->ctx->perf_frames += enc_frames;
    s->encoder_chunks++;
    s->enc_frames_total += enc_frames;
    print_text_delta(s);

    if (s->trace) {
        double audio_s = (double)s->samples_seen / (double)NEMO_SAMPLE_RATE;
        int dtok = s->ctx->perf_tokens - tokens_before;
        fprintf(stderr,
                "\n[chunk %03d] audio=%.2fs enc_frames=%d total_enc=%d tokens=+%d\n",
                s->encoder_chunks, audio_s, enc_frames, s->enc_frames_total, dtok);
    }
    return 0;
}

static int live_mel_chunk(void *user, const float *mel, int mel_frames, int final) {
    live_asr_t *s = (live_asr_t *)user;
    if (mel_frames > 0) s->mel_chunks++;
    double dec_before = s->ctx->perf_decoder_ms;
    double t0 = nemo_time_ms();
    int rc = nemo_encoder_stream_accept(s->ctx, s->enc, mel, mel_frames, final,
                                        live_encoder_chunk, s);
    double elapsed = nemo_time_ms() - t0;
    s->ctx->perf_encoder_ms += elapsed - (s->ctx->perf_decoder_ms - dec_before);
    return rc;
}

static int live_asr_init(live_asr_t *s, nemo_ctx_t *ctx, int trace) {
    memset(s, 0, sizeof(*s));
    s->ctx = ctx;
    s->trace = trace;
    s->rnnt = nemo_rnnt_stream_create(ctx);
    s->enc = nemo_encoder_stream_create(ctx);
    s->mel = nemo_mel_stream_create(ctx);
    if (!s->rnnt || !s->enc || !s->mel) return -1;
    ctx->perf_audio_ms = 0.0;
    ctx->perf_mel_ms = 0.0;
    ctx->perf_encoder_ms = 0.0;
    ctx->perf_decoder_ms = 0.0;
    ctx->perf_frames = 0;
    ctx->perf_tokens = 0;
    return 0;
}

static int live_asr_accept(live_asr_t *s, const float *samples, int n_samples, int final) {
    if (n_samples > 0) {
        s->samples_seen += n_samples;
        s->ctx->perf_audio_ms = 1000.0 * (double)s->samples_seen / (double)NEMO_SAMPLE_RATE;
    }
    double enc_before = s->ctx->perf_encoder_ms;
    double dec_before = s->ctx->perf_decoder_ms;
    double t0 = nemo_time_ms();
    int rc = nemo_mel_stream_accept(s->mel, samples, n_samples, final, live_mel_chunk, s);
    double elapsed = nemo_time_ms() - t0;
    s->ctx->perf_mel_ms += elapsed - (s->ctx->perf_encoder_ms - enc_before)
                            - (s->ctx->perf_decoder_ms - dec_before);
    return rc;
}

static void live_asr_free(live_asr_t *s) {
    if (!s) return;
    nemo_mel_stream_free(s->mel);
    nemo_encoder_stream_free(s->enc);
    nemo_rnnt_stream_free(s->rnnt);
    memset(s, 0, sizeof(*s));
}

static void audio_input_cb(void *user, AudioQueueRef queue, AudioQueueBufferRef buffer,
                           const AudioTimeStamp *start_time, UInt32 num_packets,
                           const AudioStreamPacketDescription *packet_desc) {
    mic_capture_t *cap = (mic_capture_t *)user;
    (void)start_time;
    (void)num_packets;
    (void)packet_desc;

    if (buffer->mAudioDataByteSize > 0) {
        fifo_push_i16(cap->fifo, (const int16_t *)buffer->mAudioData,
                      buffer->mAudioDataByteSize / sizeof(int16_t));
    }
    if (!cap->stopping && !g_stop) {
        AudioQueueEnqueueBuffer(queue, buffer, 0, NULL);
    }
}

static int mic_start(mic_capture_t *cap, sample_fifo_t *fifo, int buffer_ms,
                     CFStringRef device_uid, const char *device_desc) {
    memset(cap, 0, sizeof(*cap));
    cap->fifo = fifo;

    AudioStreamBasicDescription fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.mSampleRate = NEMO_SAMPLE_RATE;
    fmt.mFormatID = kAudioFormatLinearPCM;
    fmt.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
    fmt.mBytesPerPacket = sizeof(int16_t);
    fmt.mFramesPerPacket = 1;
    fmt.mBytesPerFrame = sizeof(int16_t);
    fmt.mChannelsPerFrame = 1;
    fmt.mBitsPerChannel = 16;

    OSStatus err = AudioQueueNewInput(&fmt, audio_input_cb, cap, NULL, NULL, 0, &cap->queue);
    if (err != noErr) {
        fprintf(stderr, "nemotron_mic: AudioQueueNewInput failed (%d)\n", (int)err);
        return -1;
    }
    if (device_uid) {
        err = AudioQueueSetProperty(cap->queue, kAudioQueueProperty_CurrentDevice,
                                    &device_uid, sizeof(device_uid));
        if (err != noErr) {
            fprintf(stderr, "nemotron_mic: cannot select input device %s (%d)\n",
                    device_desc && device_desc[0] ? device_desc : "(unknown)", (int)err);
            return -1;
        }
    }
    fprintf(stderr, "nemotron_mic: input device: %s\n",
            device_desc && device_desc[0] ? device_desc : "system default");

    UInt32 bytes = (UInt32)((NEMO_SAMPLE_RATE * sizeof(int16_t) * buffer_ms) / 1000);
    if (bytes < 1024) bytes = 1024;
    for (int i = 0; i < MIC_NUM_BUFFERS; i++) {
        err = AudioQueueAllocateBuffer(cap->queue, bytes, &cap->buffers[i]);
        if (err != noErr) {
            fprintf(stderr, "nemotron_mic: AudioQueueAllocateBuffer failed (%d)\n", (int)err);
            return -1;
        }
        err = AudioQueueEnqueueBuffer(cap->queue, cap->buffers[i], 0, NULL);
        if (err != noErr) {
            fprintf(stderr, "nemotron_mic: AudioQueueEnqueueBuffer failed (%d)\n", (int)err);
            return -1;
        }
    }

    err = AudioQueueStart(cap->queue, NULL);
    if (err != noErr) {
        fprintf(stderr, "nemotron_mic: AudioQueueStart failed (%d)\n", (int)err);
        return -1;
    }
    return 0;
}

static void mic_stop(mic_capture_t *cap) {
    if (!cap || !cap->queue) return;
    cap->stopping = 1;
    AudioQueueStop(cap->queue, true);
    AudioQueueDispose(cap->queue, true);
    cap->queue = NULL;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *device_arg = NULL;
    const char *lang = "auto";
    int att_right = 3;
    int push_frames = 0;
    int seconds = 0;
    int strip_tags = 0;
    int max_symbols = 10;
    int buffer_ms = 80;
    int queue_sec = 20;
    int trace = 1;
    int meter = 0;
    int list_devices = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--list-devices") == 0) {
            list_devices = 1;
        } else if ((strcmp(argv[i], "--device") == 0 || strcmp(argv[i], "-D") == 0) && i + 1 < argc) {
            device_arg = argv[++i];
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            lang = argv[++i];
        } else if (strcmp(argv[i], "--att-right") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &att_right) != 0) { usage(argv[0]); return 2; }
        } else if (strcmp(argv[i], "--push-frames") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &push_frames) != 0) { usage(argv[0]); return 2; }
        } else if (strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &seconds) != 0) { usage(argv[0]); return 2; }
        } else if (strcmp(argv[i], "--strip-tags") == 0) {
            strip_tags = 1;
        } else if (strcmp(argv[i], "--max-symbols") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &max_symbols) != 0) { usage(argv[0]); return 2; }
        } else if (strcmp(argv[i], "--buffer-ms") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &buffer_ms) != 0) { usage(argv[0]); return 2; }
        } else if (strcmp(argv[i], "--queue-sec") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &queue_sec) != 0) { usage(argv[0]); return 2; }
        } else if (strcmp(argv[i], "--meter") == 0) {
            meter = 1;
        } else if (strcmp(argv[i], "--no-trace") == 0) {
            trace = 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (list_devices) {
        print_input_devices();
        return 0;
    }

    if (!model_path || !valid_att_right(att_right) || max_symbols < 1 || buffer_ms < 10 || queue_sec < 1) {
        usage(argv[0]);
        return 2;
    }
    if (push_frames <= 0) push_frames = att_right + 1;
    if (push_frames < 1) {
        fprintf(stderr, "nemotron_mic: --push-frames must be >= 1\n");
        return 2;
    }

    nemo_ctx_t *ctx = nemo_load(model_path);
    if (!ctx) return 1;
    if (nemo_set_language(ctx, lang) != 0) {
        nemo_free(ctx);
        return 1;
    }
    ctx->att_right = att_right;
    ctx->strip_lang_tags = strip_tags;
    ctx->max_symbols_per_step = max_symbols;

    int push_samples = push_frames * NEMO_HOP_LENGTH * NEMO_SUBSAMPLING_FACTOR;
    int model_chunk_frames = ctx->att_right + 1;
    fprintf(stderr,
            "nemotron_mic: 16 kHz mono input, push=%d frames (%d ms), model_chunk=%d frames (%d ms), Ctrl-C to stop\n",
            push_frames, push_frames * 80, model_chunk_frames, model_chunk_frames * 80);

    CFStringRef device_uid = NULL;
    char device_desc[512];
    if (resolve_input_device(device_arg, &device_uid, device_desc, sizeof(device_desc)) != 0) {
        nemo_free(ctx);
        return 1;
    }

    sample_fifo_t fifo;
    if (fifo_init(&fifo, (size_t)queue_sec * NEMO_SAMPLE_RATE) != 0) {
        fprintf(stderr, "nemotron_mic: cannot allocate microphone queue\n");
        if (device_uid) CFRelease(device_uid);
        nemo_free(ctx);
        return 1;
    }

    live_asr_t asr;
    if (live_asr_init(&asr, ctx, trace) != 0) {
        fprintf(stderr, "nemotron_mic: cannot initialize streaming ASR\n");
        fifo_free(&fifo);
        if (device_uid) CFRelease(device_uid);
        nemo_free(ctx);
        return 1;
    }

    mic_capture_t cap;
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    if (mic_start(&cap, &fifo, buffer_ms, device_uid, device_desc) != 0) {
        live_asr_free(&asr);
        fifo_free(&fifo);
        if (device_uid) CFRelease(device_uid);
        nemo_free(ctx);
        return 1;
    }
    if (device_uid) {
        CFRelease(device_uid);
        device_uid = NULL;
    }

    float *chunk = (float *)malloc((size_t)push_samples * sizeof(float));
    if (!chunk) {
        g_stop = 1;
    }

    double t_start = nemo_time_ms();
    double last_meter_ms = t_start;
    size_t last_dropped = 0;
    size_t last_pushed = 0;
    int rc = 0;
    while (!g_stop) {
        if (seconds > 0 && (nemo_time_ms() - t_start) >= (double)seconds * 1000.0) break;
        if (chunk && fifo_pop_exact(&fifo, chunk, (size_t)push_samples)) {
            if (live_asr_accept(&asr, chunk, push_samples, 0) != 0) {
                rc = 1;
                break;
            }
        } else {
            sleep_ms(10);
        }
        size_t dropped = fifo_dropped(&fifo);
        if (dropped != last_dropped) {
            fprintf(stderr, "\n[nemotron_mic] dropped %.2fs of queued microphone audio\n",
                    (double)(dropped - last_dropped) / (double)NEMO_SAMPLE_RATE);
            last_dropped = dropped;
        }
        double now = nemo_time_ms();
        if (meter && now - last_meter_ms >= 1000.0) {
            size_t queued = 0;
            size_t pushed = 0;
            float peak = 0.0f;
            fifo_stats_take(&fifo, &queued, &pushed, &peak);
            fprintf(stderr, "\n[mic] captured=%.2fs (+%.2fs) queued=%.2fs peak=%.3f\n",
                    (double)pushed / (double)NEMO_SAMPLE_RATE,
                    (double)(pushed - last_pushed) / (double)NEMO_SAMPLE_RATE,
                    (double)queued / (double)NEMO_SAMPLE_RATE,
                    peak);
            last_pushed = pushed;
            last_meter_ms = now;
        }
    }

    mic_stop(&cap);
    free(chunk);

    float *tail = NULL;
    size_t tail_n = fifo_pop_all(&fifo, &tail);
    if (rc == 0) {
        rc = live_asr_accept(&asr, tail, (int)tail_n, 1) != 0;
        print_text_delta(&asr);
    }
    free(tail);

    char *text = NULL;
    if (asr.rnnt) {
        text = nemo_rnnt_stream_finish(asr.rnnt);
        asr.rnnt = NULL;
    }
    fprintf(stderr,
            "\n\nFinal: %s\n"
            "Audio: %.2fs, chunks=%d, enc_frames=%d, tokens=%d "
            "(mel %.0f ms, encoder %.0f ms, decoder %.0f ms)\n",
            text ? text : "",
            (double)asr.samples_seen / (double)NEMO_SAMPLE_RATE,
            asr.encoder_chunks,
            ctx->perf_frames,
            ctx->perf_tokens,
            ctx->perf_mel_ms,
            ctx->perf_encoder_ms,
            ctx->perf_decoder_ms);
    free(text);

    live_asr_free(&asr);
    fifo_free(&fifo);
    nemo_free(ctx);
    return rc ? 1 : 0;
}

#else

#include <stdio.h>

int main(void) {
    fprintf(stderr, "nemotron_asr_mic is currently implemented for macOS AudioQueue only.\n");
    return 1;
}

#endif
