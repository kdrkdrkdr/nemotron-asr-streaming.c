# Nemotron 3.5 ASR Streaming — Pure C

A dependency-free C inference runtime for
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).

It runs the original **cache-aware streaming** model on CPU: audio is consumed in
chunks, the FastConformer encoder keeps attention/convolution caches, and the
RNN-T decoder keeps its state across chunks — so it transcribes a live stream or
a file incrementally, in real time.

Weights are **W8A8** (packed int8, "Q8P"). Inference needs only a C toolchain —
no Python, PyTorch, NeMo, ONNX, or BLAS. (GPU is out of scope; this is a CPU
real-time streaming runtime.)

## Quick start (no Python)

```bash
make                                    # build (C toolchain only)

# download the pre-converted W8A8 model (~0.64 GiB)
huggingface-cli download kdrkdrkdr/nemotron-3.5-asr-streaming-0.6b-w8a8 \
  nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin --local-dir .

# transcribe a WAV file (16-bit PCM; resampled to 16 kHz mono if needed)
./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
  -i audio.wav -l auto --strip-tags
```

The model is `nvidia/nemotron-3.5-asr-streaming-0.6b` converted to int8. To build
the `.bin` yourself, see [Convert it yourself](#convert-it-yourself-optional).

## Real-time from a microphone

The runtime reads **raw s16le, 16 kHz, mono** from stdin via `--stdin` and prints
text as you speak. Capture the mic with `ffmpeg` and pipe it in:

```bash
ffmpeg -f avfoundation -i ":0" -ac 1 -ar 16000 -f s16le - \
  | ./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
      --stdin -l auto --strip-tags
```

Replace the `ffmpeg` input for your OS:

| OS      | ffmpeg input flags |
|---------|--------------------|
| macOS   | `-f avfoundation -i ":0"` |
| Linux   | `-f alsa -i default` |
| Windows | `-f dshow -i audio="<device>"` — list devices with `ffmpeg -list_devices true -f dshow -i dummy` |

`Ctrl-C` stops and flushes the final chunk.

> **Windows:** run the pipe under `cmd.exe` (or `cmd /c "…"`), **not** Windows
> PowerShell 5.1 — its pipeline corrupts the raw byte stream.

macOS also has a native microphone tool with device selection and a level meter:

```bash
make mic
./nemotron_asr_mic -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin -l auto --strip-tags
# --list-devices, --device X, --meter, --trace
```

## Latency vs. accuracy (`--att-right`)

The model is cache-aware with a **fixed set of chunk sizes**. `--att-right` picks
one: a bigger chunk gives more look-ahead context (better accuracy) at the cost of
more latency. One encoder frame is 80 ms.

| `--att-right` | chunk (latency) | encoder frames | use for |
|:---:|:---:|:---:|---|
| `0`  | 80 ms   | 1  | lowest latency |
| `1`  | 160 ms  | 2  | low latency |
| `3`  | 320 ms  | 4  | **default — balanced** |
| `6`  | 560 ms  | 7  | more context |
| `13` | 1120 ms | 14 | most context |

Only these five values are valid — they are the model's trained context families.

## Options

| flag | meaning |
|---|---|
| `-m <file>` | model `.bin` (required) |
| `-i <file>` | input WAV (16-bit PCM; resampled to 16 kHz mono) |
| `--stdin` | stream raw s16le 16 kHz mono from stdin (live) |
| `-l <lang>` | language prompt: `auto` (default), `en-US`, `ko-KR`, … |
| `--strip-tags` | drop emitted language tags like `<en-US>` |
| `--att-right N` | chunk size `0,1,3,6,13` (default `3`) — see table above |
| `-t <n>` | worker threads (default `1`, max 16) |
| `--max-symbols N` | max non-blank labels per frame (default 10) |
| `--model-info` | print a model summary and exit |

`-i` and `--stdin` are mutually exclusive.

## Build

```bash
make                 # optimized build (-march=native); auto-dispatches NEON / AVX2
make generic         # portable scalar fallback
make debug           # AddressSanitizer build
make check-kernels   # verify the SIMD kernels against the scalar reference
make clean
```

On Windows (an MSYS2 / MinGW shell), `make` uses `clang` and produces
`nemotron_asr.exe`. `make mic` is macOS-only. Implementation internals — kernels,
threading, Q8P packing, streaming caches, and the Win32 platform layer — are
documented in [`MODEL.md`](MODEL.md).

## Convert it yourself (optional)

Most users just download the pre-built `.bin` (see [Quick start](#quick-start-no-python)).
To build it from the original `.nemo` instead — the only step that needs Python
(`torch`, `numpy`, `yaml`):

```bash
python3 tools/convert_nemo.py path/to/nemotron-3.5-asr-streaming-0.6b.nemo \
  -o nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin --w8a8-linear-weights
```

Dense linear, LSTM, and classifier weights are written as Q8P (per-row int8 +
32-bit row scales, packed in four-row tiles); everything else stays float32.

## Library API

```c
nemo_ctx_t *ctx = nemo_load("nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin");
nemo_set_language(ctx, "ko-KR");
ctx->att_right = 3;
ctx->strip_lang_tags = 1;

char *text = nemo_transcribe(ctx, "audio.wav");   // or nemo_transcribe_stdin(ctx)
printf("%s\n", text);
free(text);
nemo_free(ctx);
```

Lower-level streaming primitives are also exposed: `nemo_mel_stream_*`,
`nemo_encoder_stream_*`, and `nemo_rnnt_stream_*`.

## License

This runtime is a format/precision conversion of NVIDIA's model and follows the
base model's [OpenMDW 1.1](https://openmdw.ai/license/1-1/) license. See the
[base model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
for intended use, languages, and limitations.
