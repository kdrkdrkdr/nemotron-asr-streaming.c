# nemotron_asr

Dependency-free pure C inference runtime for `nvidia/nemotron-3.5-asr-streaming-0.6b`.

The runtime does not load `.nemo` directly. Convert once to a simple mmap-friendly
binary, then run the C executable:

```bash
python3 tools/convert_nemo.py ../nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
  -o nemotron-3.5-asr-streaming-0.6b.bin

make
./nemotron_asr -m nemotron-3.5-asr-streaming-0.6b.bin -i ../qwen-asr/samples/jfk.wav -l en-US --strip-tags
```

Conversion requires Python with `torch` and `yaml`; inference does not.

Model facts implemented here:

- Audio: 16 kHz mono, 128 log-mel bins, `n_fft=512`, `win=400`, `hop=160`.
- Encoder: 24-layer cache-aware FastConformer, `d_model=1024`, `8 x 128` heads.
- Subsampling: causal `dw_striding`, factor 8, output feature flatten size `256 * 17`.
- Prompt: language one-hot `128` concatenated to encoder output, projected by `prompt_kernel`.
- Decoder: RNN-T greedy decoding with 2-layer LSTM prediction network and joint network.

Streaming status:

- Audio samples are accepted in chunks by the WAV convenience path.
- Mel frames are emitted incrementally once their centered analysis window is
  available, with a final flush for right-padding.
- The subsampling conv stem runs as streaming stages and emits new pre-encoder
  frames without recomputing prior stage outputs.
- The encoder runs in non-overlapping chunks of `att_right + 1` 80 ms frames,
  matching the cache-aware `att_context_size` choices from the original model.
- Each Conformer layer keeps attention K/V cache and causal convolution cache.
- The RNN-T decoder keeps prediction LSTM state and accepts encoder chunks
  incrementally.
- The current sample front-end uses a simple grow buffer for frame construction;
  replacing it with a bounded ring buffer is a memory optimization, not a graph
  semantics change.

Kernels:

- `make` builds the native dispatch path. On ARM this uses NEON; on AVX2+FMA
  x86 builds this uses AVX; otherwise it falls back to generic C.
- `make generic` forces the scalar C fallback with `NEMO_FORCE_GENERIC`.
- Kernel entry points cover the Nemotron hot path from `MODEL.md`: dense fp32
  matvec/argmax, relative-attention score dot, residual axpy, streaming preconv
  emit, and 512-point FFT power for mel features.
