# Nemotron 3.5 ASR Streaming — Pure C

A dependency-free C runtime for NVIDIA's
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
a multilingual (40 language-locales) streaming speech-to-text model.

It runs the original **cache-aware streaming** model on CPU: audio is consumed in
chunks, the FastConformer encoder keeps its attention/convolution caches, and the
RNN-T decoder keeps its state across chunks — so it transcribes a live stream or a
file incrementally, in real time. No PyTorch, no ONNX, no runtime dependencies:
the **W8A8** int8 weights (packed "Q8P") are read straight from the memory-mapped
model file, with NEON / AVX2 / scalar kernels and a small thread pool. It builds
on macOS, Linux, and natively on Windows, and does about **7× real time** on an
8-core Apple Silicon laptop.

It also adds one thing the base model doesn't have: recovery from **mid-stream
language switches** ([code-switching](#multilingual--code-switching)).

```bash
make
# raw s16le mic → live captions
ffmpeg -f avfoundation -i ":0" -ac 1 -ar 16000 -f s16le - \
  | ./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
      --stdin -l auto --strip-tags
```

## Quick start

```bash
make                                    # build (auto-dispatches NEON / AVX2)

# download the pre-converted W8A8 model (~0.62 GiB)
hf download kdrkdrkdr/nemotron-3.5-asr-streaming-0.6b-w8a8 \
  nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin --local-dir .

# transcribe a WAV file (16-bit PCM; resampled to 16 kHz mono if needed)
./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
  -i audio.wav -l auto --strip-tags
```

`hf` is the Hugging Face CLI (`pip install huggingface_hub`; the older
`huggingface-cli download …` works too). The `.bin` is `nemotron-3.5-asr-streaming-0.6b`
converted to int8 — see [Convert it yourself](#convert-it-yourself-optional) to
rebuild it from the original `.nemo`.

## Real-time from a microphone

The runtime reads **raw s16le, 16 kHz, mono** from stdin via `--stdin` and prints
text as you speak. Capture the mic with `ffmpeg` and pipe it in:

```bash
ffmpeg -f avfoundation -i ":0" -ac 1 -ar 16000 -f s16le - \
  | ./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
      --stdin -l auto --strip-tags --att-right 3 -t 4
```

`--att-right` sets the live latency — lower is snappier, higher waits for more
look-ahead context (`3` = 320 ms, the default). See the
[latency table](#latency-vs-accuracy---att-right). `Ctrl-C` stops and flushes the
final chunk.

Replace the `ffmpeg` input for your OS:

| OS      | ffmpeg input flags |
|---------|--------------------|
| macOS   | `-f avfoundation -i ":0"` |
| Linux   | `-f alsa -i default` |
| Windows | `-f dshow -i audio="<device>"` — list devices with `ffmpeg -list_devices true -f dshow -i dummy` |

> **Windows:** run the pipe under `cmd.exe` (or `cmd /c "…"`), **not** Windows
> PowerShell 5.1 — its pipeline corrupts the raw byte stream.

> **macOS:** the terminal app needs microphone permission
> (System Settings → Privacy & Security → Microphone).

## Multilingual & code-switching

The model handles 40 language-locales from one set of weights, selected by a
language prompt (`-l en-US`, `-l ko-KR`, …, or `-l auto` to auto-detect and emit
a `<xx-YY>` tag). This runtime adds recovery from a **mid-stream language switch**
— say English, then Korean — which the base model drops.

When a speaker switches language, the onset of the switched-to language (often
the first ~3 s) would otherwise vanish. Streaming cache-aware RNN-T carries state
momentum across the switch: the prediction network (an internal LM) still
"speaks" the previous language and the encoder's attention cache still holds its
context, so the new onset decodes as a long run of blanks and is lost. This is
architectural, not a port bug — single-utterance and continuous same-language
speech match ground truth exactly, and the drop appears only at a switch.

The runtime recovers by watching for the decoder's **own** stall: a run of
~1.28 s (`NEMO_STALL_FRAMES`, default 16 encoder frames × 80 ms) of all-blank
output means it has lost the thread. It then re-primes the RNN-T prediction
network to its blank/start state and clears the encoder's per-layer attention
K/V and convolution caches, so decoding re-locks onto the current audio. The
trigger is the model's blank output — **not** audio-energy VAD, so there is no dB
threshold and it is language-agnostic — and it is gated on "have we emitted
anything yet," so leading silence never clips the first word and genuine silence
stays untouched. On by default; tune or disable with `NEMO_STALL_FRAMES` (`0`
disables).

**Verified against the original model.** The upstream `.nemo` was run in NeMo
itself on the same real-speech clips (Google FLEURS), in both its full-context
offline mode *and* its cache-aware streaming mode. It drops the second language
on two of these three switches in either mode; this runtime reproduces that
exactly with recovery off (a faithful port — with recovery off it matches the
NeMo streaming output onset-for-onset) and recovers all three with it on:

| switch (real speech) | upstream NeMo (offline & streaming) | this runtime, off | this runtime, on |
|---|:---:|:---:|:---:|
| single language (en, ko) | ✅ | ✅ | ✅ |
| English → Korean | ✅ | ✅ | ✅ |
| Korean → English | ❌ English dropped | ❌ | ✅ **recovered** |
| Spanish → Japanese | ❌ Japanese dropped | ❌ | ✅ **recovered** |

At the lowest latencies the base model drops even more: streaming NeMo at
`--att-right 1` also loses the Korean onset of English → Korean — again matching
this runtime with recovery off, and again recovered with it on. Validated across
eight cross-language pairs (en/ko/es/fr/de/ja/zh/ru), both directions, offline
and live, with no regression on single-language, continuous, or short-alternation
audio.

Two honest caveats: it is a decode-time heuristic (not retraining) — it improves
a case the base model wasn't designed for, and does **not** turn a streaming
RNN-T into a non-streaming model like Whisper; and a clip that ends on a breath
during a long pause can leave one trailing filler token in file mode (harmless).

## Latency vs. accuracy (`--att-right`)

The model is cache-aware with a **fixed set of chunk sizes**. `--att-right` picks
one: a bigger chunk gives more look-ahead context (better accuracy) at the cost of
more latency. One encoder frame is 80 ms.

| `--att-right` | chunk (latency) | encoder frames | use for |
|:---:|:---:|:---:|---|
| `0`  | 80 ms   | 1  | lowest latency |
| `1`  | 160 ms  | 2  | low latency |
| `3`  | 320 ms  | 4  | **default — balanced** |
| `6`  | 560 ms  | 7  | most context |

Only these four values are valid.

## Options

| flag | meaning |
|---|---|
| `-m <file>` | model `.bin` (required) |
| `-i <file>` | input WAV (16-bit PCM; resampled to 16 kHz mono) |
| `--stdin` | stream raw s16le 16 kHz mono from stdin (live) |
| `-l <lang>` | language prompt: `auto` (default), `en-US`, `ko-KR`, … |
| `--strip-tags` | drop emitted language tags like `<en-US>` |
| `--att-right N` | chunk size `0,1,3,6` (default `3`) — see table above |
| `-t <n>` | worker threads (default `1`, max 16) |
| `--max-symbols N` | max non-blank labels per frame (default 10) |
| `--model-info` | print a model summary and exit |

`-i` and `--stdin` are mutually exclusive. The `NEMO_STALL_FRAMES` environment
variable tunes code-switch recovery (default `16` frames; `0` disables it).

## Build

```bash
make                 # optimized build (-march=native); auto-dispatches NEON / AVX2
make generic         # portable scalar fallback (-DNEMO_FORCE_GENERIC)
make debug           # -O0 -g (+ AddressSanitizer on POSIX)
make clean
```

On Windows (an MSYS2 / MinGW shell), `make` uses `clang` and produces
`nemotron_asr.exe`, with a small Win32 layer replacing pthread/mmap.

## How it works

```text
audio samples
  → streaming log-mel frames (n_fft 512, win 400, hop 160, 128 mel bins)
  → causal depthwise-striding subsampling (factor 8)
  → 24 FastConformer layers  (rel-pos attention + conv, per-layer K/V + conv cache)
  → language-prompt fusion
  → RNN-T greedy decoder (2-layer LSTM prediction state kept across chunks)
  → text
```

The dominant cost is dense matvec, stored and computed as packed **W8A8 Q8P**
(per-row int8 + 32-bit row scales, four-output-row tiles, consumed directly from
the mmap'd file). Backends: scalar (`_generic`), baseline NEON (`vmull_s8` +
`vpadalq_s16`), and AVX2 (`madd_epi16`) — no dotprod/i8mm/VNNI required.
Threading is a persistent pool with output-row splitting. Full internals —
kernels, Q8P packing, streaming caches, and the Win32 platform layer — are in
[`MODEL.md`](MODEL.md).

## Library API

```c
nemo_ctx_t *ctx = nemo_load("nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin");
nemo_set_language(ctx, "ko-KR");   /* or "auto" */
ctx->att_right = 3;
ctx->strip_lang_tags = 1;

char *text = nemo_transcribe(ctx, "audio.wav");   /* or nemo_transcribe_stdin(ctx) */
printf("%s\n", text);
free(text);
nemo_free(ctx);
```

Lower-level streaming primitives are also exposed: `nemo_mel_stream_*`,
`nemo_encoder_stream_*`, and `nemo_rnnt_stream_*`. When you drive the encoder and
RNN-T streams yourself, forward the decoder's code-switch signal to the encoder
each chunk:

```c
if (nemo_rnnt_stream_take_enc_reset(rnnt))
    nemo_encoder_stream_request_reset(enc);
```

## Convert it yourself (optional)

Most users just download the pre-built `.bin` (see [Quick start](#quick-start)).
To rebuild it from the original `.nemo`, grab the converter from the model repo
([`convert_nemo.py`](https://huggingface.co/kdrkdrkdr/nemotron-3.5-asr-streaming-0.6b-w8a8/blob/main/convert_nemo.py))
— it needs Python (`torch`, `numpy`, `yaml`):

```bash
python3 convert_nemo.py path/to/nemotron-3.5-asr-streaming-0.6b.nemo \
  -o nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin --w8a8-linear-weights
```

Dense linear, LSTM, and classifier weights are written as Q8P (per-row int8 +
32-bit row scales, four-row tiles); everything else stays float32.

## License

This runtime is a format/precision conversion of NVIDIA's model and follows the
base model's [OpenMDW 1.1](https://openmdw.ai/license/1-1/) license. See the
[base model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
for intended use, languages, and limitations.
