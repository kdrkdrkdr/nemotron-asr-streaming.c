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

The first version favors clarity and correctness of the graph over speed. The hottest
future work is replacing the naive DFT and generic matmuls with pure-C FFT/SIMD kernels.
