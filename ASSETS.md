# Embedded data and models

go-g2p ships everything it needs inside the module and embeds it in your binary,
so `g2p.New` and `neural.New` need no file paths. The core package embeds about
19 MB (dictionaries, tokenizer data, POS model). `fallback/neural` embeds another
6 MB, only when you import it.

| Files | Size | Source | License |
|---|---|---|---|
| `data/us_gold.json, us_silver.json, gb_gold.json, gb_silver.json` | 12.6 MB | Misaki's dictionaries at commit `fba1236`, unchanged | Apache-2.0 |
| `data/tokenizer.json` | 375.3 KB | spaCy `en_core_web_sm` 3.8.0 tokenizer rules and norms, exported as data | MIT |
| `data/unicode.json` | 12.7 KB | Unicode 15.0 digit values (from Python 3.12's `unicodedata`) | Unicode License |
| `models/pos.onnx, pos.json` | 6.3 MB | `en_core_web_sm` 3.8.0 trained tagger, converted to ONNX ([notice](models/README.md)) | MIT |
| `fallback/neural/models/*` | 6.3 MB | PeterReid's US/UK models, converted to ONNX ([notice](fallback/neural/models/README.md)) | Apache-2.0 |

Every file can be regenerated from its pinned source with [`tools/export`](tools/export),
which also checks that the result matches the committed bytes.

## Overriding the embedded files

`Config.DataDir` replaces all six `data/` files, `Config.ModelDir` replaces
`pos.json`/`pos.onnx`, and `neural.Config.ModelDir` replaces the six neural files.
Each directory must contain every file it replaces, with the same names.

## What is not included

- **ONNX Runtime.** Install it yourself; select it with `ORTLibrary` or own the
  environment in your application (see the README). The current Go binding
  requests C API 22; native 1.22.0 is the tested baseline.
- **eSpeak-ng**, for `fallback/espeak` only (tested Ubuntu eSpeak-ng 1.51).
  `espeak.Config.Executable` selects its path.

## Provenance limits

The neural model cards declare Apache-2.0 but document no training data. Misaki's
dictionaries are distributed in its Apache-2.0 repository without a separate
statement of where their entries came from. See THIRD_PARTY.md.
