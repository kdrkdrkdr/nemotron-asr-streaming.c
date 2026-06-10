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

CPU engine optimizations now mirror the useful qwen-asr patterns that apply to
Nemotron's graph:

- a persistent worker pool controlled by `nemo_set_threads`
- output-row parallelism for large fp32 `nemo_linear` calls
- threaded `nemo_matvec_f32` and `nemo_argmax_matvec_f32`
- optional BF16 model conversion for dense linear, LSTM, and joint classifier
  weights, consumed directly without expanding a full float32 copy
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

Qwen-only kernels that are not used by the Nemotron graph should not be carried
over just for symmetry.

References used while matching the graph:

- NVIDIA NeMo `ConformerEncoder` source: https://docs.nvidia.com/nemo/speech/nightly/_modules/nemo/collections/asr/modules/conformer_encoder.html
- NVIDIA NeMo `ConvSubsampling` source: https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/subsampling.py
- NVIDIA NeMo RNN-T modules: https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/rnnt.py
