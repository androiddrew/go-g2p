# Attribution and asset terms

Original code is Apache-2.0 (LICENSE/NOTICE). Adapted behavior retains upstream
attribution; the retained notices are under `third_party/licenses/`.

| Component | Source/version | Terms |
|---|---|---|
| Misaki frontend and fallback conversion behavior | hexgrad/misaki `fba1236595f2d2bf21d414ba6e57d25256afada3` | Apache-2.0, `misaki.txt` |
| Tokenizer/POS behavior | spaCy 3.8.7, Thinc 8.3.11 | MIT, `spacy-LICENSE.txt`, `thinc-LICENSE.txt` |
| Trained English model (embedded as `data/tokenizer.json`, `models/pos.*`) | en_core_web_sm 3.8.0 | MIT plus training-source notices, `en-core-web-sm-*.txt` |
| Misaki dictionaries (embedded as `data/*_gold.json`, `data/*_silver.json`) | hexgrad/misaki `fba1236595f2d2bf21d414ba6e57d25256afada3`, unchanged | Apache-2.0, `misaki.txt` |
| Neural fallback models (embedded as `fallback/neural/models/*`) | PeterReid/graphemes_to_phonemes_en_us `a5631b2…`, _en_gb `d8357d5…`; modified (ONNX conversion) | Apache-2.0; changes in `fallback/neural/models/README.md` |
| Unicode digit table | Unicode 15.0 | `unicode.txt` |
| regexp2 | v1.11.5 | MIT plus upstream attributions |
| golang.org/x/text | v0.28.0 | BSD-3-Clause, `golang-x-text.txt` |
| onnxruntime_go | v1.22.0 | External Go dependency, MIT |
| ortenv | github.com/androiddrew/ortenv v0.1.0 | MIT; Copyright (c) 2026 Drew Bednar |

The dictionary data are distributed in the Apache-2.0 Misaki repository; no
separate file-level dictionary-origin/license statement was established. Neural
US/UK model cards declare Apache-2.0 with sparse training-data provenance. The
embedded models and dictionaries retain their source terms.
Arithmetic number spelling is independently implemented Go code.

The optional `fallback/espeak` package uses external eSpeak-ng, tested with Ubuntu
`1.51+dfsg-12build1`. Its executable/data remain **GPL-3.0-or-later** with the
corresponding license/source distribution obligations; separate execution does
not relicense them. Sources: https://github.com/espeak-ng/espeak-ng/tree/1.51 and
the matching Ubuntu source package. The package contains no eSpeak code and does
not link libespeak-ng: it runs the `espeak-ng` executable as a separate process.
Programs that do not import `fallback/espeak` have no eSpeak implementation,
library, executable or data dependency.

Misaki reaches eSpeak through bootphon/phonemizer (GPL-3.0). `fallback/espeak`
does not include or translate phonemizer code. To produce the same phonemes as
Misaki, it reproduces two phonemizer behaviors that Misaki enables:
`preserve_punctuation`, using phonemizer's documented default set of 21
punctuation marks (`;:,.!?¡¿—…"«»“”(){}[]`), and `language_switch='remove-flags'`,
which strips eSpeak's `(lang)` switch markers. Its phoneme mapping table adapts
Misaki's Apache-2.0 `EspeakFallback`, as noted above.

Static test fixtures under `testdata/` record expected Misaki/spaCy outputs as
JSON; they contain no code from, and do not run, another implementation.

The native ONNX Runtime is installed independently by the user and is not shipped
in this repository. Retained notices above apply to adapted code/data and selected
source dependencies; they are not a license bundle for a system runtime install.
