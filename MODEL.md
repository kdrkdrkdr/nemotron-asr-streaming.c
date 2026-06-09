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

References used while matching the graph:

- NVIDIA NeMo `ConformerEncoder` source: https://docs.nvidia.com/nemo/speech/nightly/_modules/nemo/collections/asr/modules/conformer_encoder.html
- NVIDIA NeMo `ConvSubsampling` source: https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/subsampling.py
- NVIDIA NeMo RNN-T modules: https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/rnnt.py
