# Nemotron 3.5 ASR Model Notes

Local `.nemo` contents:

- `model_config.yaml`
- `model_weights.ckpt`
- SentencePiece tokenizer model and vocab text

The checkpoint has 657 float32 tensors. Important groups:

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
The runtime supports those dense weights as either float32 or direct BF16.

Secondary targets:

- RNNT joint argmax: about 0.335B MAC for the smoke sample
- pre-encode pointwise conv and projection: about 0.253B MAC
- prediction LSTM: about 0.098B MAC for initial state plus 14 emitted tokens

Current SIMD backends should therefore prioritize:

- `nemo_matvec_f32` and the dot helper it uses
- `nemo_argmax_matvec_f32` for the RNNT joint classifier
- `nemo_attention_score_f32`, because it is a tiny but deeply nested loop
- `nemo_vec_axpy_inplace` for residual and attention value accumulation
- `nemo_preconv_emit_f32` for streaming causal subsampling conv stages
- `nemo_fft512_power_f32` plus `nemo_dot_f32` for the mel front-end
- typed dense wrappers such as `nemo_linear_weight` and
  `nemo_argmax_matvec_weight` for direct BF16 linear weights
- BF16 dense helpers: `nemo_dot_bf16_f32_impl`,
  `nemo_bf16_matvec_fused_impl`, and `nemo_argmax_bf16_range_impl`
- W8A8 dense helpers: row-major `nemo_q8_matvec_fused_impl` /
  `nemo_argmax_q8_range_impl` for compatibility, plus packed
  `nemo_q8p_matvec_fused_impl` / `nemo_argmax_q8p_range_impl` for new
  W8A8 model files

CPU engine optimizations now mirror the useful qwen-asr patterns that apply to
Nemotron's graph:

- a persistent worker pool controlled by `nemo_set_threads`
- output-row parallelism for large fp32 `nemo_linear` calls
- threaded `nemo_matvec_f32` and `nemo_argmax_matvec_f32`
- optional BF16 model conversion for dense linear, LSTM, and joint classifier
  weights, consumed directly without expanding a full float32 copy
- experimental W8A8 model conversion for the same dense weights, stored as
  packed Q8P: per-row int8 plus float32 row scales in four-output-row tiles
  with a 16-byte padded input stride, consumed directly from the mmap'd model
  file
- architecture-dispatched BF16 row dot, matvec, and classifier argmax range
  for those typed dense weights
- architecture-dispatched Q8P matvec and classifier argmax range for
  linear-only W8A8 experiments, with row-major Q8 retained as a compatibility
  loader/runtime path
- fused `nemo_linear3_nobias` for encoder Q/K/V projection
- threaded `nemo_prompt_linear_relu` for language prompt projection
- fused `nemo_lstm_gates_f32` for RNN-T prediction-network gate projection
- persistent per-stream encoder, prompt, and RNN-T projection scratch buffers
  to avoid chunk-local malloc/free
- optional BLAS dispatch for larger dense batches, while tiny streaming chunks
  stay on the native kernels

The fused Q/K/V path matters because streaming chunks are small. Running three
separate threaded linear calls pays thread-dispatch overhead three times per
layer. Fusing keeps the same math while scheduling one pass over
`rows * d_model` output positions.

The LSTM gate fusion applies the same idea to the prediction network: the
input and recurrent projections are accumulated in one output-row pass instead
of launching two matvecs and a separate residual add for every emitted token.

The BF16 path is intentionally applied only to dense weights that are consumed
by the typed linear/matvec wrappers. Biases, layer-norm parameters, depthwise
convolution filters, mel front-end tensors, and the prediction embedding remain
float32.

The W8A8 path follows the same boundary. It quantizes dense linear-family
weights offline with one scale per output row, then packs them into Q8P tiles:
four output rows at a time, input dimension padded to a 16-byte stride. At
runtime, each input vector is quantized symmetrically to int8 immediately
before the typed matvec, padded with zeros to that stride, accumulated into
int32, and dequantized back to float32. The rest of the graph stays float32.
This keeps the original streaming model structure intact while making the
dominant dense operations cheaper to load and compute.

Backend-specific BF16 coverage:

- generic C: reference BF16 row dot, matvec, and classifier argmax range
- NEON: BF16 row dot plus two-output-row matvec and classifier argmax range
- AVX2/FMA: BF16 row dot plus two-output-row matvec and classifier argmax range
- AVX512F+BW: 16-lane BF16 conversion plus four-output-row matvec and
  classifier argmax range

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
- Use shared activation prequantization only for single-row Q8/Q8P calls, where
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
- Q8/Q8P runtime wrappers keep the temporary activation quantization buffer on
  the stack for input vectors up to 4096 elements, with heap fallback for
  larger inputs. This covers the dense Nemotron shapes while avoiding
  malloc/free in the streaming hot path.

The row grouping matters for this model because dense operations are mostly
streaming matvecs. Reusing each loaded input vector across two or four output
rows reduces instruction overhead and memory traffic pressure without changing
the graph.

Qwen-only kernels that are not used by the Nemotron graph should not be carried
over just for symmetry.

References used while matching the graph:

- NVIDIA NeMo `ConformerEncoder` source: https://docs.nvidia.com/nemo/speech/nightly/_modules/nemo/collections/asr/modules/conformer_encoder.html
- NVIDIA NeMo `ConvSubsampling` source: https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/subsampling.py
- NVIDIA NeMo RNN-T modules: https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/rnnt.py
