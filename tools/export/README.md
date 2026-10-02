# Export tooling

`export.py` rebuilds every file go-g2p embeds, from pinned sources:

- Misaki's US/UK dictionaries at commit `fba1236`, copied unchanged.
- spaCy `en_core_web_sm` 3.8.0: the tokenizer rules (`data/tokenizer.json`) and
  the trained POS tagger (`models/pos.onnx`, `models/pos.json`).
- Python 3.12's Unicode 15.0 digit table (`data/unicode.json`).
- PeterReid's US/UK grapheme-to-phoneme models on Hugging Face, converted to ONNX
  (`fallback/neural/models/`).

Every download is checked against a SHA-256 pinned in the script. The Go module
does not need Python; this is only for auditing or regenerating the files.

```bash
cd tools/export
uv run --frozen python export.py --output out --check
```

`--check` compares each output with the committed file and fails on any
difference. With the locked environment in `uv.lock` (Python 3.12, torch
2.6.0+cpu, transformers 4.51.3, spaCy 3.8.7, Thinc 8.3.11, onnx 1.17.0) on
Linux/amd64, all 14 outputs reproduce byte-for-byte. Copy `out/` over the
repository only when intentionally changing the embedded files.
