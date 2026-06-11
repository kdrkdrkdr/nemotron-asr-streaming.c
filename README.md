# Nemotron 3.5 ASR Streaming Pure C

This is a dependency-free C inference runtime for
`nvidia/nemotron-3.5-asr-streaming-0.6b`.

The runtime is designed around the original cache-aware streaming model:
audio is accepted in chunks, mel frames are emitted incrementally, the
FastConformer encoder keeps attention and convolution caches, and the RNN-T
decoder keeps prediction-network state across chunks.

Inference does not require Python, PyTorch, NeMo, YAML, SentencePiece, BLAS, or
any external runtime library. The `.nemo` model is converted once into a simple
mmap-friendly `.bin` file.

## Supported Modes

- **WAV mode**: transcribe a WAV file through the same streaming pipeline used
  by live input.
- **Live microphone mode**: macOS AudioQueue input, selectable input device,
  chunk trace output, and audio level metering.
- **Model conversion**: extract `model_config.yaml`, `model_weights.ckpt`, and
  the RNN-T vocabulary from the original `.nemo` archive into the C runtime
  binary format.

This project focuses on dependency-free CPU inference. There is no CUDA, MPS,
PyTorch, ONNX Runtime, or NeMo dependency in the inference path.

## Quick Start

```bash
# Convert the original NeMo archive once.
python3 tools/convert_nemo.py \
  ../nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
  -o nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
  --w8a8-linear-weights

# Build the normal WAV CLI.
make

# Transcribe a WAV file.
./nemotron_asr \
  -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
  -i ../qwen-asr/samples/jfk.wav \
  -l en-US \
  --strip-tags
```

Conversion requires Python with `torch`, `numpy`, and `yaml`. Inference does not.

For this branch, use the W8A8 linear file. Dense linear, LSTM, and classifier
weights are stored as packed Q8P int8 with 32-bit row scales. Runtime
activation vectors are quantized at the typed dense call sites, then the graph
continues in its normal streaming representation.

## Features

- **Pure C inference**: C11 runtime with libc/libm/pthread only by default.
- **Memory-mapped model file**: converted weights are mmap'd from a compact
  tensor stream.
- **Packed W8A8 linear weights**: primary converter path stores dense linear,
  LSTM, and vocabulary classifier weights as Q8P packed int8 and dynamically
  quantizes activations for int8 matvec inference.
- **Original streaming shape**: chunk size is controlled by Nemotron's
  cache-aware `att_context_size` right context.
- **Incremental front end**: audio samples and mel frames are accepted
  incrementally.
- **Streaming encoder**: causal subsampling, per-layer attention K/V cache, and
  causal convolution cache.
- **Streaming RNN-T decoder**: prediction LSTM state is preserved across
  encoder chunks.
- **Language prompt control**: supports prompt IDs such as `auto`, `en-US`, and
  `ko-KR`.
- **Native kernels**: generic C fallback plus NEON and AVX2 dispatch for the
  W8A8 hot path.
- **Threaded dense path**: qwen-asr-style persistent worker pool for large
  dense linear, matvec, and classifier operations.
- **Fused QKV projection**: encoder attention computes Q/K/V in one scheduled
  projection pass to reduce dispatch overhead on small streaming chunks.
- **Low-allocation streaming loop**: encoder, prompt, and RNN-T projection
  scratch buffers are reused across chunks.
- **Live testing**: macOS microphone CLI with input device selection and a
  real-time input meter.

## Build

```bash
make          # native optimized build
make generic  # force scalar C fallback with NEMO_FORCE_GENERIC
make debug    # AddressSanitizer debug build
make mic      # build macOS live microphone tool
make check-kernels  # compare native W8A8 kernels against generic C
make bench-kernels  # microbenchmark generic/native W8A8 matvec/argmax paths
make check-arch-syntax  # syntax-check NEON and AVX kernel variants
make clean
```

`make` uses `-march=native`, `-ffast-math`, and LTO by default. On ARM builds it
dispatches to NEON kernels. On AVX2+FMA x86 builds it dispatches to AVX kernels.
Otherwise it uses the generic C backend. `make check-arch-syntax` is a
clang-oriented cross-syntax check for the architecture-specific kernel files.
`make bench-kernels` measures generic and native backend speed for
representative dense matvec and classifier argmax shapes such as FFN,
attention, and joint vocabulary projections. It also reports the packed Q8P
runtime wrapper path that includes dynamic activation quantization.

## Convert Model

The C runtime does not load `.nemo` directly. Convert once into the W8A8 model
file used by this branch:

```bash
python3 tools/convert_nemo.py \
  ../nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
  -o nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
  --w8a8-linear-weights
```

The converter reads:

- `model_config.yaml`
- `model_weights.ckpt`
- `cfg["joint"]["vocabulary"]`

It writes a little-endian `.bin` file containing 64-byte-aligned tensor payloads
followed by vocabulary strings. With `--w8a8-linear-weights`, dense encoder,
prompt, RNN-T prediction, and joint classifier weights are written as Q8P:
per-output-row int8 plus 32-bit row scales, packed in four-output-row tiles
with a 16-byte padded input stride. Runtime activation vectors are quantized to
int8 only at typed linear/matvec call sites, then padded with zeros to the
packed stride.

## WAV Usage

```bash
./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin -i audio.wav
```

Useful options:

```bash
./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin -i audio.wav -l ko-KR
./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin -i audio.wav -l en-US --strip-tags
./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin -i audio.wav --att-right 6
./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin -i audio.wav -t 8
./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin --model-info
```

The WAV path loads the file, resamples to 16 kHz mono if needed, and then feeds
the audio through the same chunked streaming graph used by live input.

## Which Mode To Use

- **Quick file test**: use `./nemotron_asr -i audio.wav` with the default
  `--att-right 3`.
- **Lowest interactive latency**: use the microphone tool with `--att-right 0`
  or `--att-right 1`. This emits smaller encoder chunks.
- **Balanced live transcription**: use `--att-right 3`, the model's default
  320 ms chunk family.
- **More right context**: use `--att-right 6` or `--att-right 13`. This adds
  latency but can improve difficult audio.
- **CPU throughput testing**: use the W8A8 linear model file and set `-t N` to
  the number of useful CPU cores.

## Live Microphone

Live microphone input is currently implemented for macOS through AudioQueue.
By default, recognized text is written continuously to stdout as it is emitted.
Chunk and token diagnostics are hidden unless `--trace` is enabled.

```bash
make mic

./nemotron_asr_mic \
  -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
  -l ko-KR \
  --strip-tags \
  --att-right 3
```

List input devices:

```bash
./nemotron_asr_mic --list-devices
```

Select a device by index, AudioDeviceID, UID, or exact name:

```bash
./nemotron_asr_mic \
  -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
  --device 1 \
  -t 8 \
  -l ko-KR \
  --strip-tags \
  --att-right 3
```

On macOS, virtual devices such as BlackHole or VB-Cable may be the default
input. If microphone audio seems silent, run `--list-devices` and pass the
actual microphone with `--device`.

For input debugging, add `--meter`. Before transcript output begins it shows a
temporary status line with captured audio diagnostics:

```text
[mic] captured=3.00s (+1.00s) queued=0.12s peak=0.083
```

- `captured` not increasing usually means permission or capture callback
  trouble.
- `captured` increasing with `peak=0.000` usually means the selected input is
  silent.
- `peak` moving means audio is reaching the runtime.

Use `Ctrl-C` to stop and flush the final chunk.

For chunk-level debugging:

```bash
./nemotron_asr_mic -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin --trace
```

## Streaming Chunk Settings

`--att-right` selects the original cache-aware chunk family. One encoder frame
is 80 ms after factor-8 subsampling.

| `--att-right` | encoder frames per chunk | chunk duration |
|---------------|--------------------------|----------------|
| `0`           | `1`                      | 80 ms          |
| `1`           | `2`                      | 160 ms         |
| `3`           | `4`                      | 320 ms         |
| `6`           | `7`                      | 560 ms         |
| `13`          | `14`                     | 1120 ms        |

Default:

```bash
--att-right 3
```

The microphone tool also accepts `--push-frames N`. This controls how often
captured samples are pushed into the streaming graph, in 80 ms encoder-frame
units. If omitted, it defaults to `att_right + 1`.

```bash
# Lower input push latency while keeping the model chunk family.
./nemotron_asr_mic -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
  --device 1 --att-right 3 --push-frames 1 --meter
```

The model still emits encoder chunks according to `--att-right`; `--push-frames`
only controls how frequently microphone samples are handed to the front end.

## How Streaming Works

This runtime follows Nemotron's cache-aware FastConformer-RNN-T design instead
of using text rollback.

Pipeline:

```text
audio samples
-> streaming log-mel frames
-> causal subsampling conv stages
-> FastConformer chunks with attention K/V cache and conv cache
-> RNN-T greedy decoder with persistent prediction LSTM state
-> text
```

Important details:

- Mel frames are emitted as soon as their centered analysis window is available.
- The subsampling conv stem emits only new pre-encoder frames.
- Encoder chunks are non-overlapping and have `att_right + 1` frames.
- Encoder attention reuses cached left context from previous chunks.
- Conformer convolution modules keep causal GLU history.
- RNN-T decoding keeps prediction-network hidden and cell state.

## Language

Default language prompt is `auto`.

Examples:

```bash
./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin -i audio.wav -l auto
./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin -i audio.wav -l en-US
./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin -i audio.wav -l ko-KR
```

Use `--strip-tags` to remove emitted language tags such as `<en-US>`.

## Kernels

The public kernel surface focuses on the current W8A8 hot path:

- packed Q8P matvec
- packed Q8P classifier argmax range
- typed-weight linear wrappers for direct Q8P dispatch
- fused encoder Q/K/V projection
- language prompt projection
- fused RNN-T prediction LSTM gates
- supporting streaming kernels for attention, residual accumulation,
  subsampling preconv, and mel front-end work

Backends:

- `nemotron_asr_kernels_generic.c`: scalar C fallback for W8A8 Q8P dense paths.
- `nemotron_asr_kernels_neon.c`: ARM NEON implementation for base integer SIMD
  Q8P matvec and classifier argmax.
- `nemotron_asr_kernels_avx.c`: AVX2 implementation for Q8P matvec and
  classifier argmax.
- Q8P paths use baseline NEON and AVX2 integer SIMD only: no dotprod, i8mm,
  VNNI, or AVX512-VNNI dependency is required.

The normal convolution, layer norm, softmax, activation, and model-binding
utilities live in the shared runtime code.

Threading:

- `nemo_set_threads()` controls a persistent worker pool, capped at 16 threads.
- The CLI defaults to all online CPUs and accepts `-t N`.
- Large dense calls are split across output rows.
- Small operations stay single-threaded to avoid scheduling overhead.
- W8A8 linear weights are consumed directly from the mmap'd packed int8 payload
  and use dynamic per-vector activation quantization.
- Single-row W8A8 calls share one prequantized activation buffer across output
  worker threads; multi-row encoder calls keep per-worker stack quantization to
  avoid an extra synchronization point.
- Q8P matvec and classifier argmax use four-output-row packed int8 tiles with a
  16-byte padded stride in the generic, NEON, and AVX2 backends.

Example JFK sample timing on an 8-core Apple Silicon laptop:

| build/options | inference | realtime |
|---------------|-----------|----------|
| native W8A8 Q8P linear, `-t 8` | 1.57 s | 7.03x |

## Smoke Checks

```bash
python3 -m py_compile tools/convert_nemo.py
make clean && make
make check-kernels
make bench-kernels
make check-arch-syntax

./nemotron_asr \
  -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
  -i ../qwen-asr/samples/jfk.wav \
  -l en-US \
  --strip-tags \
  -t 8

make generic
./nemotron_asr \
  -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
  -i ../qwen-asr/samples/jfk.wav \
  -l en-US \
  --strip-tags \
  -t 8
```

## Model Facts

- Audio: 16 kHz mono.
- Features: 128 log-mel bins, `n_fft=512`, `win=400`, `hop=160`.
- Encoder: 24-layer cache-aware FastConformer.
- Encoder width: `d_model=1024`.
- Attention: 8 heads, 128 dimensions per head.
- Subsampling: causal `dw_striding`, factor 8.
- Pre-encoder flatten size: `256 * 17`.
- Prompt: language one-hot of size 128, projected into encoder output.
- Decoder: RNN-T greedy decoding with 2-layer LSTM prediction network.
- Joint: encoder projection, prediction projection, ReLU, vocabulary classifier.

See `MODEL.md` for the tensor mapping and kernel-priority analysis.

## API Sketch

```c
nemo_ctx_t *ctx = nemo_load("nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin");
nemo_set_language(ctx, "ko-KR");
ctx->att_right = 3;
ctx->strip_lang_tags = 1;

char *text = nemo_transcribe(ctx, "audio.wav");
free(text);
nemo_free(ctx);
```

Lower-level streaming pieces are also exposed:

- `nemo_mel_stream_create/accept/free`
- `nemo_encoder_stream_create/accept/free`
- `nemo_rnnt_stream_create/accept/finish/free`
- `nemo_rnnt_stream_text`
- `nemo_rnnt_stream_text_len`

## Troubleshooting

### Microphone is silent

```bash
./nemotron_asr_mic --list-devices
./nemotron_asr_mic -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin --device 1 --meter
```

Check macOS microphone permission for the app you launch from, such as Terminal,
iTerm, or VS Code.

### Output starts slowly

Use a smaller chunk family:

```bash
--att-right 0
--att-right 1
```

Smaller chunks reduce latency but may reduce accuracy.

### Output quality is unstable

Try the default chunk family first:

```bash
--att-right 3
```

For more right context:

```bash
--att-right 6
--att-right 13
```

Larger chunks add latency but usually give the encoder more context.
