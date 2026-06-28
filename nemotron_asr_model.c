/*
 * nemotron_asr_model.c - model file loader.
 * Memory-maps the converted .bin (mmap on POSIX, CreateFileMapping on Windows),
 * parses the tensor table and vocabulary, and binds named tensors (F32 and
 * packed Q8P int8) into the encoder/decoder/joint structures.
 */
#include "nemotron_asr.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define NEMO_MODEL_MAGIC "NM35ASR"
#define NEMO_MODEL_VERSION 1u

static uint64_t rd_u64(const unsigned char **p) {
    uint64_t v = 0;
    memcpy(&v, *p, sizeof(v));
    *p += sizeof(v);
    return v;
}

static uint32_t rd_u32(const unsigned char **p) {
    uint32_t v = 0;
    memcpy(&v, *p, sizeof(v));
    *p += sizeof(v);
    return v;
}

static uint16_t rd_u16(const unsigned char **p) {
    uint16_t v = 0;
    memcpy(&v, *p, sizeof(v));
    *p += sizeof(v);
    return v;
}

static const unsigned char *align64(const unsigned char *base, const unsigned char *p) {
    uintptr_t off = (uintptr_t)(p - base);
    off = (off + 63u) & ~(uintptr_t)63u;
    return base + off;
}

static uint64_t align_u64(uint64_t v, uint64_t a) {
    return (v + a - 1u) & ~(a - 1u);
}

static uint64_t tensor_row_elems(uint8_t ndims, const uint64_t dims[4]) {
    uint64_t n = 1;
    for (uint8_t d = 1; d < ndims; d++) n *= dims[d];
    return n;
}

static char *dup_bytes(const unsigned char *p, size_t n) {
    char *s = (char *)malloc(n + 1);
    if (!s) return NULL;
    memcpy(s, p, n);
    s[n] = 0;
    return s;
}

const nemo_tensor_t *nemo_model_find(const nemo_model_t *model, const char *name) {
    for (int i = 0; i < model->n_tensors; i++) {
        if (strcmp(model->tensors[i].name, name) == 0) return &model->tensors[i];
    }
    return NULL;
}

static int bind_tensor(const nemo_model_t *m, const char *name, const float **dst) {
    const nemo_tensor_t *t = nemo_model_find(m, name);
    if (!t) {
        fprintf(stderr, "nemotron: missing tensor %s\n", name);
        return -1;
    }
    if (t->dtype != NEMO_TENSOR_F32 || !t->data) {
        fprintf(stderr, "nemotron: tensor %s is not float32\n", name);
        return -1;
    }
    *dst = t->data;
    return 0;
}

static int bind_weight(const nemo_model_t *m, const char *name, nemo_weight_t *dst) {
    const nemo_tensor_t *t = nemo_model_find(m, name);
    if (!t) {
        fprintf(stderr, "nemotron: missing tensor %s\n", name);
        return -1;
    }
    memset(dst, 0, sizeof(*dst));
    dst->dtype = t->dtype;
    if (t->dtype == NEMO_TENSOR_Q8P && t->data_q8 && t->q8_scales) {
        dst->q8 = t->data_q8;
        dst->q8_scales = t->q8_scales;
        dst->q8_stride = (uint32_t)t->q8_stride;
        return 0;
    }
    fprintf(stderr, "nemotron: tensor %s is not Q8P (dtype %u)\n", name, t->dtype);
    return -1;
}

static int bind_layer(nemo_model_t *m, int i) {
    char key[256];
    nemo_enc_layer_t *l = &m->encoder.layers[i];
#define BIND_TENSOR(field, suffix) do { \
    snprintf(key, sizeof(key), "encoder.layers.%d.%s", i, suffix); \
    if (bind_tensor(m, key, &l->field) != 0) return -1; \
} while (0)
#define BIND_WEIGHT(field, suffix) do { \
    snprintf(key, sizeof(key), "encoder.layers.%d.%s", i, suffix); \
    if (bind_weight(m, key, &l->field) != 0) return -1; \
} while (0)
    BIND_TENSOR(norm_ff1_w, "norm_feed_forward1.weight");
    BIND_TENSOR(norm_ff1_b, "norm_feed_forward1.bias");
    BIND_WEIGHT(ff1_linear1_w, "feed_forward1.linear1.weight");
    BIND_WEIGHT(ff1_linear2_w, "feed_forward1.linear2.weight");
    BIND_TENSOR(norm_conv_w, "norm_conv.weight");
    BIND_TENSOR(norm_conv_b, "norm_conv.bias");
    BIND_WEIGHT(conv_pw1_w, "conv.pointwise_conv1.weight");
    BIND_TENSOR(conv_dw_w, "conv.depthwise_conv.weight");
    BIND_TENSOR(conv_norm_w, "conv.batch_norm.weight");
    BIND_TENSOR(conv_norm_b, "conv.batch_norm.bias");
    BIND_WEIGHT(conv_pw2_w, "conv.pointwise_conv2.weight");
    BIND_TENSOR(norm_att_w, "norm_self_att.weight");
    BIND_TENSOR(norm_att_b, "norm_self_att.bias");
    BIND_TENSOR(pos_bias_u, "self_attn.pos_bias_u");
    BIND_TENSOR(pos_bias_v, "self_attn.pos_bias_v");
    BIND_WEIGHT(att_q_w, "self_attn.linear_q.weight");
    BIND_WEIGHT(att_k_w, "self_attn.linear_k.weight");
    BIND_WEIGHT(att_v_w, "self_attn.linear_v.weight");
    BIND_WEIGHT(att_out_w, "self_attn.linear_out.weight");
    BIND_WEIGHT(att_pos_w, "self_attn.linear_pos.weight");
    BIND_TENSOR(norm_ff2_w, "norm_feed_forward2.weight");
    BIND_TENSOR(norm_ff2_b, "norm_feed_forward2.bias");
    BIND_WEIGHT(ff2_linear1_w, "feed_forward2.linear1.weight");
    BIND_WEIGHT(ff2_linear2_w, "feed_forward2.linear2.weight");
    BIND_TENSOR(norm_out_w, "norm_out.weight");
    BIND_TENSOR(norm_out_b, "norm_out.bias");
#undef BIND_TENSOR
#undef BIND_WEIGHT
    return 0;
}

static int bind_known_tensors(nemo_model_t *m) {
    nemo_encoder_t *e = &m->encoder;
    nemo_decoder_t *d = &m->decoder;
    nemo_joint_t *j = &m->joint;
    if (bind_tensor(m, "preprocessor.featurizer.window", &e->window) != 0) return -1;
    if (bind_tensor(m, "preprocessor.featurizer.fb", &e->mel_fb) != 0) return -1;
    if (bind_tensor(m, "encoder.pre_encode.conv.0.weight", &e->pre_conv0_w) != 0) return -1;
    if (bind_tensor(m, "encoder.pre_encode.conv.0.bias", &e->pre_conv0_b) != 0) return -1;
    if (bind_tensor(m, "encoder.pre_encode.conv.2.weight", &e->pre_conv2_w) != 0) return -1;
    if (bind_tensor(m, "encoder.pre_encode.conv.2.bias", &e->pre_conv2_b) != 0) return -1;
    if (bind_tensor(m, "encoder.pre_encode.conv.3.weight", &e->pre_conv3_w) != 0) return -1;
    if (bind_tensor(m, "encoder.pre_encode.conv.3.bias", &e->pre_conv3_b) != 0) return -1;
    if (bind_tensor(m, "encoder.pre_encode.conv.5.weight", &e->pre_conv5_w) != 0) return -1;
    if (bind_tensor(m, "encoder.pre_encode.conv.5.bias", &e->pre_conv5_b) != 0) return -1;
    if (bind_tensor(m, "encoder.pre_encode.conv.6.weight", &e->pre_conv6_w) != 0) return -1;
    if (bind_tensor(m, "encoder.pre_encode.conv.6.bias", &e->pre_conv6_b) != 0) return -1;
    if (bind_weight(m, "encoder.pre_encode.out.weight", &e->pre_out_w) != 0) return -1;
    if (bind_tensor(m, "encoder.pre_encode.out.bias", &e->pre_out_b) != 0) return -1;
    for (int i = 0; i < NEMO_ENC_LAYERS; i++) if (bind_layer(m, i) != 0) return -1;
    if (bind_weight(m, "prompt_kernel.0.weight", &e->prompt0_w) != 0) return -1;
    if (bind_tensor(m, "prompt_kernel.0.bias", &e->prompt0_b) != 0) return -1;
    if (bind_weight(m, "prompt_kernel.2.weight", &e->prompt2_w) != 0) return -1;
    if (bind_tensor(m, "prompt_kernel.2.bias", &e->prompt2_b) != 0) return -1;

    if (bind_tensor(m, "decoder.prediction.embed.weight", &d->embed_w) != 0) return -1;
    for (int l = 0; l < NEMO_PRED_LAYERS; l++) {
        char key[128];
        snprintf(key, sizeof(key), "decoder.prediction.dec_rnn.lstm.weight_ih_l%d", l);
        if (bind_weight(m, key, &d->lstm_w_ih[l]) != 0) return -1;
        snprintf(key, sizeof(key), "decoder.prediction.dec_rnn.lstm.weight_hh_l%d", l);
        if (bind_weight(m, key, &d->lstm_w_hh[l]) != 0) return -1;
        snprintf(key, sizeof(key), "decoder.prediction.dec_rnn.lstm.bias_ih_l%d", l);
        if (bind_tensor(m, key, &d->lstm_b_ih[l]) != 0) return -1;
        snprintf(key, sizeof(key), "decoder.prediction.dec_rnn.lstm.bias_hh_l%d", l);
        if (bind_tensor(m, key, &d->lstm_b_hh[l]) != 0) return -1;
    }

    if (bind_weight(m, "joint.pred.weight", &j->pred_w) != 0) return -1;
    if (bind_tensor(m, "joint.pred.bias", &j->pred_b) != 0) return -1;
    if (bind_weight(m, "joint.enc.weight", &j->enc_w) != 0) return -1;
    if (bind_tensor(m, "joint.enc.bias", &j->enc_b) != 0) return -1;
    if (bind_weight(m, "joint.joint_net.2.weight", &j->out_w) != 0) return -1;
    if (bind_tensor(m, "joint.joint_net.2.bias", &j->out_b) != 0) return -1;
    return 0;
}

int nemo_model_load(nemo_model_t *model, const char *path) {
    memset(model, 0, sizeof(*model));
#ifdef _WIN32
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "nemotron: cannot open %s (error %lu)\n", path, GetLastError());
        return -1;
    }
    LARGE_INTEGER fsz;
    if (!GetFileSizeEx(fh, &fsz) || fsz.QuadPart <= 0) {
        fprintf(stderr, "nemotron: cannot stat %s\n", path);
        CloseHandle(fh);
        return -1;
    }
    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) {
        fprintf(stderr, "nemotron: file mapping failed for %s (error %lu)\n", path, GetLastError());
        CloseHandle(fh);
        return -1;
    }
    void *map = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!map) {
        fprintf(stderr, "nemotron: map view failed for %s (error %lu)\n", path, GetLastError());
        CloseHandle(mh);
        CloseHandle(fh);
        return -1;
    }
    model->win_file = (void *)fh;
    model->win_mapping = (void *)mh;
    model->map = map;
    model->map_size = (size_t)fsz.QuadPart;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "nemotron: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "nemotron: cannot stat %s\n", path);
        close(fd);
        return -1;
    }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        fprintf(stderr, "nemotron: mmap failed for %s: %s\n", path, strerror(errno));
        return -1;
    }
    model->map = map;
    model->map_size = (size_t)st.st_size;
#endif

    const unsigned char *base = (const unsigned char *)map;
    const unsigned char *p = base;
    const unsigned char *end = base + model->map_size;
    if ((size_t)(end - p) < 16 || memcmp(p, NEMO_MODEL_MAGIC, 7) != 0) {
        fprintf(stderr, "nemotron: %s is not a nemotron-asr model bin\n", path);
        nemo_model_free(model);
        return -1;
    }
    p += 8;
    uint32_t version = rd_u32(&p);
    if (version != NEMO_MODEL_VERSION) {
        fprintf(stderr, "nemotron: unsupported model version %u\n", version);
        nemo_model_free(model);
        return -1;
    }
    uint32_t n_tensors = rd_u32(&p);
    uint32_t vocab_size = rd_u32(&p);
    (void)rd_u32(&p);
    model->n_tensors = (int)n_tensors;
    model->tensors = (nemo_tensor_t *)calloc(n_tensors, sizeof(nemo_tensor_t));
    if (!model->tensors) {
        nemo_model_free(model);
        return -1;
    }

    for (uint32_t ti = 0; ti < n_tensors; ti++) {
        if (p + 2 > end) { nemo_model_free(model); return -1; }
        uint16_t name_len = rd_u16(&p);
        uint8_t ndims = *p++;
        uint8_t dtype = *p++;
        if ((dtype != NEMO_TENSOR_F32 && dtype != NEMO_TENSOR_Q8P) ||
            ndims > 4 || p + name_len + 4 * 8 + 8 > end) {
            fprintf(stderr, "nemotron: corrupt tensor table\n");
            nemo_model_free(model);
            return -1;
        }
        char *name = dup_bytes(p, name_len);
        p += name_len;
        for (int d = 0; d < 4; d++) model->tensors[ti].dims[d] = rd_u64(&p);
        uint64_t nbytes = rd_u64(&p);
        p = align64(base, p);
        if (p + nbytes > end) {
            fprintf(stderr, "nemotron: tensor %s overruns file\n", name ? name : "?");
            free(name);
            nemo_model_free(model);
            return -1;
        }
        model->tensors[ti].name = name;
        model->tensors[ti].dtype = dtype;
        model->tensors[ti].ndims = ndims;
        model->tensors[ti].data = dtype == NEMO_TENSOR_F32 ? (const float *)p : NULL;
        if (dtype == NEMO_TENSOR_Q8P) {
            uint64_t rows = model->tensors[ti].dims[0];
            uint64_t cols = tensor_row_elems(ndims, model->tensors[ti].dims);
            uint64_t scale_bytes = rows * sizeof(float);
            if (rows == 0 || cols == 0 || scale_bytes > nbytes) {
                fprintf(stderr, "nemotron: corrupt q8p tensor %s\n", name ? name : "?");
                nemo_model_free(model);
                return -1;
            }
            uint64_t stride = align_u64(cols, 16);
            uint64_t packed_rows = align_u64(rows, 4);
            uint64_t data_bytes = packed_rows * stride;
            model->tensors[ti].q8_stride = stride;
            if (scale_bytes > nbytes || data_bytes > nbytes - scale_bytes) {
                fprintf(stderr, "nemotron: corrupt q8p tensor %s\n", name ? name : "?");
                nemo_model_free(model);
                return -1;
            }
            model->tensors[ti].q8_scales = (const float *)p;
            model->tensors[ti].data_q8 = (const int8_t *)(p + scale_bytes);
        }
        model->tensors[ti].nbytes = nbytes;
        p += nbytes;
        p = align64(base, p);
    }

    if (p + 4 > end) { nemo_model_free(model); return -1; }
    uint32_t file_vocab = rd_u32(&p);
    if (file_vocab != vocab_size) {
        fprintf(stderr, "nemotron: vocab size mismatch\n");
        nemo_model_free(model);
        return -1;
    }
    model->vocab_size = (int)vocab_size;
    model->vocab = (char **)calloc(vocab_size, sizeof(char *));
    if (!model->vocab) {
        nemo_model_free(model);
        return -1;
    }
    for (uint32_t i = 0; i < vocab_size; i++) {
        if (p + 4 > end) { nemo_model_free(model); return -1; }
        uint32_t len = rd_u32(&p);
        if (p + len > end) { nemo_model_free(model); return -1; }
        model->vocab[i] = dup_bytes(p, len);
        p += len;
        if (!model->vocab[i]) {
            nemo_model_free(model);
            return -1;
        }
    }

    if (bind_known_tensors(model) != 0) {
        nemo_model_free(model);
        return -1;
    }
    return 0;
}

void nemo_model_free(nemo_model_t *model) {
    if (!model) return;
    if (model->vocab) {
        for (int i = 0; i < model->vocab_size; i++) free(model->vocab[i]);
        free(model->vocab);
    }
    if (model->tensors) {
        for (int i = 0; i < model->n_tensors; i++) free((void *)model->tensors[i].name);
        free(model->tensors);
    }
#ifdef _WIN32
    if (model->map) UnmapViewOfFile(model->map);
    if (model->win_mapping) CloseHandle((HANDLE)model->win_mapping);
    if (model->win_file) CloseHandle((HANDLE)model->win_file);
#else
    if (model->map && model->map != MAP_FAILED) munmap(model->map, model->map_size);
#endif
    memset(model, 0, sizeof(*model));
}
