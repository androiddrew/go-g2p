# Neural fallback models

These files are **modified versions** of the Apache-2.0 grapheme-to-phoneme
models published by the Hugging Face user PeterReid and used by Misaki's neural fallback:

| Files | Source |
|---|---|
| `us-encoder.onnx`, `us-decoder.onnx`, `us.json` | https://huggingface.co/PeterReid/graphemes_to_phonemes_en_us at revision `a5631b285d18d59483c32c0c3379cb9fac924f4b` |
| `gb-encoder.onnx`, `gb-decoder.onnx`, `gb.json` | https://huggingface.co/PeterReid/graphemes_to_phonemes_en_gb at revision `d8357d5067fa26a5c34134d6bbcf4bbf000c0ac8` |

Changes made for go-g2p (Apache-2.0 §4(b) notice): each BART model was split into
an encoder and a single-step decoder and converted from PyTorch safetensors to
ONNX (opset 17, float32). The weights were not retrained or altered. The JSON
files restate the models' character tables and generation limits from their
`config.json` and `generation_config.json`. The model cards declare
`license: apache-2.0` and document no training data.

Regenerate and verify with `../../../tools/export` (see its README).
