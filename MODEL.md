# Nemotron 3.5 ASR Model Notes

Local `.nemo` contents:

- `model_config.yaml`
- `model_weights.ckpt`
- SentencePiece tokenizer model and vocab text

The checkpoint has 657 tensors. Important groups:

- `preprocessor.featurizer.window`: `[400]`
- `preprocessor.featurizer.fb`: `[1,128,257]`
- `encoder.pre_encode.*`: causal depthwise striding subsampler
- `encoder.layers.{0..23}.*`: FastConformer blocks
- `decoder.prediction.*`: RNNT prediction LSTM
- `joint.*`: RNNT joint projections and vocab classifier
- `prompt_kernel.*`: language prompt fusion MLP

Architecture from `model_config.yaml`:

- Sample rate: `16000`
- Mel features: `128`, `window_size=0.025`, `window_stride=0.01`, `n_fft=512`
- Encoder: ConformerEncoder, `n_layers=24`, `d_model=1024`, `n_heads=8`
- Encoder attention: relative position, chunked limited context, default `[56,3]`
- Encoder conv module: kernel `9`, causal context `[8,0]`, layer norm
- Decoder: RNNTDecoder, 2 LSTM layers, hidden `640`, `blank_as_pad=true`
- Joint: hidden `640`, output `13087 + blank`
- Prompt: `128` language IDs; `auto` is prompt id `101`

Streaming chunk choices are controlled by the right context:

- `[56,0]`: 80 ms
- `[56,1]`: 160 ms
- `[56,3]`: 320 ms
- `[56,6]`: 560 ms
- `[56,13]`: 1120 ms

In the original model this is a cache-aware streaming setup:

- chunk size is `att_right + 1` encoder frames, where one encoder frame is 80 ms
  after the stride-8 subsampling stack
- chunks are processed without overlap
- the causal stride subsampling stack can emit new encoder frames incrementally
- each encoder layer keeps self-attention cache and convolution cache
- the RNN-T prediction network keeps its recurrent state across chunks

## Kernel Priority Analysis

The kernel plan is based on the converted model tensor shapes, not copied from
Qwen. For the 2 second smoke sample the runtime emits 26 encoder frames after
the stride-8 subsampling stack. MAC counts below use that `T=26` frame count.

Per encoder layer:

- FFN1 linear1 `1024 -> 4096`: 109.1M MAC
- FFN1 linear2 `4096 -> 1024`: 109.1M MAC
- self-attention Q/K/V/out projections: 109.1M MAC
- FFN2 linear1 `1024 -> 4096`: 109.1M MAC
- FFN2 linear2 `4096 -> 1024`: 109.1M MAC
- convolution pointwise1 `1024 -> 2048`: 54.5M MAC
- relative-position projection `1024 -> 1024` over `2T-1`: 53.5M MAC
- convolution pointwise2 `1024 -> 1024`: 27.3M MAC
- attention score dot and value accumulation: about 1.2M MAC total

Across 24 layers this is about 16.37B MAC for the encoder stack. The dominant
hot path is therefore dense matvec/linear, including FFN, attention
projections, conformer pointwise conv, pre-encode projection, and joint argmax.
This branch stores and consumes those dense weights as packed W8A8 Q8P.

Secondary targets:

- RNNT joint argmax: about 0.335B MAC for the smoke sample
- pre-encode pointwise conv and projection: about 0.253B MAC
- prediction LSTM: about 0.098B MAC for initial state plus 14 emitted tokens

Current SIMD backends should therefore prioritize:

- packed Q8P matvec for dense encoder, prompt, prediction, and joint weights
- packed Q8P classifier argmax for the RNNT joint output
- fused Q/K/V projection, because streaming chunks are small and launch
  overhead matters
- fused prediction LSTM gate projection for emitted tokens
- supporting streaming kernels for attention, residual accumulation, causal
  subsampling, and the mel front-end

CPU engine optimizations now mirror the useful qwen-asr patterns that apply to
Nemotron's graph:

- a persistent worker pool controlled by `nemo_set_threads`
- output-row parallelism for large dense linear calls
- threaded dense matvec and classifier argmax calls
- W8A8 model conversion for dense linear, LSTM, and joint classifier weights,
  stored as packed Q8P: per-row int8 plus 32-bit row scales in
  four-output-row tiles with a 16-byte padded input stride, consumed directly
  from the mmap'd model file
- architecture-dispatched Q8P matvec and classifier argmax range for W8A8
  inference
- fused `nemo_linear3_nobias` for encoder Q/K/V projection
- threaded `nemo_prompt_linear_relu` for language prompt projection
- fused RNN-T prediction-network gate projection
- persistent per-stream encoder, prompt, and RNN-T projection scratch buffers
  to avoid chunk-local malloc/free

The fused Q/K/V path matters because streaming chunks are small. Running three
separate threaded linear calls pays thread-dispatch overhead three times per
layer. Fusing keeps the same math while scheduling one pass over
`rows * d_model` output positions.

The LSTM gate fusion applies the same idea to the prediction network: the
input and recurrent projections are accumulated in one output-row pass instead
of launching two matvecs and a separate residual add for every emitted token.

The W8A8 path intentionally applies only to dense weights consumed by the typed
linear/matvec wrappers. Biases, layer-norm parameters, depthwise convolution
filters, mel front-end tensors, and the prediction embedding keep their
original representation.

It quantizes dense linear-family weights offline with one scale per output row,
then packs them into Q8P tiles: four output rows at a time, input dimension
padded to a 16-byte stride. At runtime, each input vector is quantized
symmetrically to int8 immediately before the typed matvec, padded with zeros to
that stride, accumulated into int32, and dequantized back into the normal
runtime output buffers. This keeps the original streaming model structure
intact while making the dominant dense operations cheaper to load and compute.

Backend-specific W8A8 coverage:

- generic C: scalar int8 dot plus four-output-row Q8P matvec and classifier
  argmax tiles
- NEON: base integer SIMD path using signed int8 multiply, int16 products, and
  int32 accumulation, with four-output-row Q8P matvec and classifier argmax
  tiles
- AVX2/FMA: signed int8 dot via int8->int16 widening and `madd_epi16`, with
  four-output-row Q8P matvec and classifier argmax tiles

W8A8 tuning notes from the JFK smoke path:

- Keep multi-row encoder activation quantization inside the output-row
  workers. A broad shared prequantization pass removes duplicate work, but it
  serializes a step that was previously parallel and regressed the 8-thread
  path.
- Use shared activation prequantization only for single-row Q8P calls, where
  the same input vector would otherwise be quantized once per output worker.
- Keep prompt/LSTM scalar row helpers out of the main encoder hot path unless
  the replacement is measured end to end. A wrapper-dispatched row-dot path was
  slower on the smoke path despite using the same quantized row-dot boundary.
- W8A8 int8 kernels intentionally stay on baseline architecture SIMD:
  base NEON uses `vmull_s8` plus `vpadalq_s16`, and AVX2 uses signed
  int8->int16 widening plus `madd_epi16`. The branch does not require NEON
  dotprod/i8mm or x86 VNNI.
- Scalar activation quantization is currently kept because Clang's optimized
  scalar loop was faster end to end than a hand-written NEON quantizer in the
  streaming benchmark.
- Q8P weight packing is now done by the converter. The runtime reads the
  packed mmap payload directly and only zero-pads the temporary activation
  vector to the tensor stride.
- The NEON Q8P four-output-row tile intentionally stays at one 16-byte int8
  multiply/accumulate step per loop to keep register pressure modest.
- An eight-output-row Q8P tile was tested and rejected on Apple Silicon because
  the extra accumulators hurt the compiler/backend enough to regress the JFK
  smoke path.
- Q8P runtime wrappers keep the temporary activation quantization buffer on
  the stack for input vectors up to 4096 elements, with heap fallback for
  larger inputs. This covers the dense Nemotron shapes while avoiding
  malloc/free in the streaming hot path.

The row grouping matters for this model because dense operations are mostly
streaming matvecs. Reusing each loaded input vector across two or four output
rows reduces instruction overhead and memory traffic pressure without changing
the graph.

Qwen-only kernels that are not used by the Nemotron graph should not be carried
over just for symmetry.

## C Runtime Streaming Pipeline

The runtime follows Nemotron's cache-aware FastConformer-RNN-T design instead of
text rollback:

```text
audio samples
-> streaming log-mel frames
-> causal subsampling conv stages
-> FastConformer chunks with attention K/V cache and conv cache
-> RNN-T greedy decoder with persistent prediction LSTM state
-> text
```

Emission details: mel frames are emitted as soon as their centered analysis
window is available; the subsampling conv stem emits only new pre-encoder
frames; encoder chunks are non-overlapping with `att_right + 1` frames and reuse
cached left context; conformer conv keeps causal GLU history; the RNN-T decoder
keeps prediction-network hidden and cell state across chunks.

## Code-switch Recovery

The base model is single-language-per-utterance: `auto` detects the language and
appends a `<xx-YY>` tag after the terminal punctuation. Its streaming decode has
no per-switch state reset, so a mid-utterance language switch drops the onset of
the switched-to language. The cause is carried-over streaming state: after the
switch the RNN-T prediction network (an internal LM) still favors the previous
language, and the encoder's attention cache still holds its left context, so the
new onset decodes as a long run of blanks. This is a property of the model, not
the port — single-utterance and continuous same-language decode match the
reference exactly (see the comparison below).

The runtime recovers with a small, non-VAD heuristic driven by the model's own
output. In `nemo_rnnt_stream_accept` a per-stream `blank_run` counts consecutive
all-blank encoder frames. Once it reaches `NEMO_STALL_RESET_FRAMES` (16 frames =
~1.28 s, overridable per stream by the `NEMO_STALL_FRAMES` env var; `0` disables)
and at least one token has been emitted (`emitted_any`, so leading silence never
clips the first word), the decoder:

- re-primes the prediction network to its start state (zero hidden/cell, a
  blank `pred_step`, and a fresh `pred_proj`), and
- raises `enc_reset_req` on its own stream.

The signal is stream-local, never on the shared context. The orchestration reads
it after each chunk with `nemo_rnnt_stream_take_enc_reset` and forwards it via
`nemo_encoder_stream_request_reset`, which sets the encoder stream's
`reset_pending`; the next chunk clears that stream's per-layer attention K/V and
conv cache lengths before encoding. Because the encoder produces a chunk and then
decodes it (in lock-step through the callback) before the next chunk, the flag
set on chunk *k* is consumed before chunk *k+1* encodes. The `%` condition
re-fires every `NEMO_STALL_FRAMES` through a long stall — a single reset recovers
the near onset but not a longer, encoder-side stall. Genuine silence is safe: a
re-primed decoder still emits blanks for it (measured: zero spurious tokens over
a 5 s silent gap).

### Verification against the reference

To confirm the drop is the base model's behavior, the original
`nemotron-3.5-asr-streaming-0.6b.nemo` was run in NeMo (`EncDecRNNTBPEModelWithPrompt`,
`target_lang="auto"`) on real speech (Google FLEURS clips concatenated into
cross-language switches), in both full-context offline `transcribe()` and
cache-aware streaming (`conformer_stream_step` with `CacheAwareStreamingAudioBuffer`,
`att_context_size = [56, att_right]`). The reference drops the second language on
Korean→English and Spanish→Japanese in both modes, and additionally loses the
Korean onset of English→Korean in streaming at `att_right 1`. This runtime with
recovery off reproduces those drops onset-for-onset (faithful port); with recovery
on it keeps all of them, with no regression on single-language, continuous, or
short-alternation audio, across eight language pairs in both directions.

## Public Kernel Surface

The exposed kernels focus on the W8A8 hot path:

- packed Q8P matvec
- packed Q8P classifier argmax range
- typed-weight linear wrappers for direct Q8P dispatch
- fused encoder Q/K/V projection
- language prompt projection
- fused RNN-T prediction LSTM gates
- supporting streaming kernels for attention, residual accumulation,
  subsampling preconv, and mel front-end work

Convolution, layer norm, softmax, activation, and model-binding utilities live
in the shared runtime code. Backends: `nemotron_asr_kernels_generic.c` (scalar
fallback), `nemotron_asr_kernels_neon.c` (baseline NEON), and
`nemotron_asr_kernels_avx.c` (AVX2) — Q8P paths use baseline integer SIMD only
(no dotprod/i8mm/VNNI). Threading uses `nemo_set_threads()` (persistent pool,
capped at 16) with output-row splitting for large dense calls.

Example JFK sample timing on an 8-core Apple Silicon laptop: native W8A8 Q8P
linear, `-t 8` — 1.57 s inference, 7.03x realtime.

## Windows Platform Layer

The runtime builds natively on Windows from an MSYS2/MinGW shell: the Makefile
detects the Windows shell and builds with `clang` (MSVC-runtime target by
default) under `-march=native`, producing `nemotron_asr.exe`. A small Win32
layer replaces the POSIX dependencies — the thread pool uses `SRWLOCK` plus
condition variables instead of pthreads, the model file is mapped with
`CreateFileMapping`/`MapViewOfFile` instead of `mmap`, and CPU count and timing
use Win32 APIs. AVX2+FMA dispatch is unchanged because clang defines
`__AVX2__`/`__FMA__` under `-march=native` even on the MSVC target.

References used while matching the graph:

- NVIDIA NeMo `ConformerEncoder` source: https://docs.nvidia.com/nemo/speech/nightly/_modules/nemo/collections/asr/modules/conformer_encoder.html
- NVIDIA NeMo `ConvSubsampling` source: https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/subsampling.py
- NVIDIA NeMo RNN-T modules: https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/rnnt.py
