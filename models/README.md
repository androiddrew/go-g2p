# POS tagger model

`pos.onnx` is the trained part-of-speech tagger from spaCy's `en_core_web_sm`
3.8.0 (MIT; see `../third_party/licenses/en-core-web-sm-LICENSE.txt`), converted
to ONNX (opset 17, float32) for go-g2p. The trained weights are copied unchanged;
only the computation was re-expressed so ONNX Runtime can run it. `pos.json`
holds the tag labels and feature-hashing parameters from the same model.

Changed from the original: format conversion only. Regenerate and verify with
`tools/export` (see its README).
