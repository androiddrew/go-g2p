# go-g2p

`github.com/androiddrew/go-g2p` (package `g2p`) is a reusable **US/UK English
text-to-phoneme frontend**. It includes normalization, dictionaries, morphology,
stress, inline overrides and trained tokenizer/POS behavior. Its Go runtime has
no dependency on a speech synthesizer or interpreter. All dictionaries and models
are embedded, so the only external requirement is an installed ONNX Runtime.

```go
import (
    "context"
    "errors"
    "fmt"
    g2p "github.com/androiddrew/go-g2p"
    "github.com/androiddrew/go-g2p/fallback/neural"
)

func phonemize(ctx context.Context) (err error) {
    const ortLibrary = "libonnxruntime.so" // OS loader name or explicit installed-library path
    fb, err := neural.New(neural.Config{ORTLibrary: ortLibrary})
    if err != nil { return err }
    defer func() { err = errors.Join(err, fb.Close()) }() // closes after the engine

    engine, err := g2p.New(g2p.Config{
        ORTLibrary: ortLibrary,
        Fallback:   fb,
    })
    if err != nil { return err }
    defer func() { err = errors.Join(err, engine.Close()) }()

    result, err := engine.Phonemize(ctx, g2p.Request{Text: "Hello, world!", Dialect: g2p.US}) // or g2p.GB
    if err != nil { return err }
    fmt.Println(result.Phonemes)
    return nil
}
```

## Pronunciation fallback

Words missing from the dictionaries go to `Config.Fallback`, any `fallback.Engine`.
Choose one at runtime; none is linked into your program unless you import it.

| Package | Backend | Needs | License of what it runs |
|---|---|---|---|
| `fallback/neural` | Misaki's US/UK neural models via ONNX Runtime | nothing (models embedded, +6 MB) | Apache-2.0 models |
| `fallback/espeak` | External `espeak-ng` executable | eSpeak-ng installed | eSpeak-ng is GPL-3.0-or-later |
| `nil` | None: unknown words become `unresolved` diagnostics | nothing | — |

`fallback/espeak` contains no eSpeak code and never links eSpeak; it starts the
program as a separate process. A program that does not import it has no GPL
dependency. If you distribute the `espeak-ng` executable with your application,
its GPL terms apply to that executable. The G2P engine never closes the fallback:
close it after the engine. You can also supply your own `fallback.Engine`.

`Result` contains phonemes, tokens, pronunciation sources, fallback calls,
diagnostics and completeness. Detected unresolved or generation-limited spans return
`IncompleteError` together with the partial result. `AllowTruncated` is an explicit
diagnostic opt-in; `Complete` remains false. `Request.Debug` adds tagger internals
(`Result.Debug`) and neural token IDs. Overrides such as `[Astra](/ˈæstɹə/)` are supported.

## ONNX Runtime ownership

ONNX Runtime has one environment per process, shared by every model in it. The
trained POS tagger always uses it, as does `fallback/neural`.

- **Set `ORTLibrary`** (simplest): the engine initializes the environment through
  [`github.com/androiddrew/ortenv`](https://github.com/androiddrew/ortenv) on first
  use. The environment stays loaded until process exit, so closing the last engine
  never unloads the runtime. Every component using ortenv in the process must use
  the same library selector, and nothing may destroy the environment ortenv owns.
- **Leave `ORTLibrary` empty** when your application already manages ONNX Runtime
  itself (directly through `onnxruntime_go` or through ortenv). Initialize the
  environment before `New` and destroy it only after `Close`; otherwise `New`
  returns `ErrNotInitialized`. Use the same choice for `neural.Config.ORTLibrary`.

`Config.Threads` and `neural.Config.Threads` set intra-op threads per session
(default 1). With many models in one process, keep these small to avoid
oversubscribing the CPU.

## Runtime and behavior

Close every engine. Each engine serializes its requests and Close. Native Run
cannot be interrupted mid-call. Both dialects are loaded eagerly. Requires Go
1.25 or newer with CGO. Tested: Linux/amd64, Go 1.25.0 and 1.26.4, ORT 1.22.0;
POS and neural fallback execute on CPU. This is a known tested configuration, not
an exact native-version restriction. Runtime selection is user-managed and must be
compatible with the resolved Go binding's C API.

Known behavior limits: clock strings such as `10:30` may be pronounced incorrectly;
spell out the time or override it. Names, URLs and invented words need review.
Neural unresolved spans are limited to 62 code points and 21 generated tokens.
Numeric expansion is bounded by uint64 magnitudes. Whole requests are retained
in memory; text is bounded to 100,000 code points. Completeness is not a universal
pronunciation-accuracy guarantee.

## Development and tools

```bash
go test ./...      # eSpeak tests skip unless espeak-ng is installed
go vet ./...
go build ./cmd/g2p-probe   # --fallback neural|espeak|none
```

For native tests, set `G2P_TEST_ORT` to an ONNX Runtime library path or loader
name, then run `go test -race ./...`. With `espeak-ng`
installed, the same run also checks the eSpeak backend. The tests compare
tokenizer, POS and full-pipeline output with expected results committed under
`testdata/` (recorded from Misaki and spaCy). `G2P_TEST_TOKENIZER` can point the
tokenizer test at a different fixture file. Native tests skip when `G2P_TEST_ORT`
is unset. Tests that need ONNX Runtime uninitialized each run in a fresh
subprocess, because ortenv keeps the environment loaded until process exit.

[`tools/export`](tools/export) regenerates the embedded files from their pinned
sources and checks them against the committed bytes (Python, not needed to use
the library).

`cmd/g2p-probe` accepts JSON requests on stdin and emits JSON results;
`cmd/fallback-probe` inspects direct fallback; `cmd/pos-probe` checks trained POS
using `testdata/pos.json`. These are optional Go-native diagnostic tools.

See [ASSETS.md](ASSETS.md), [THIRD_PARTY.md](THIRD_PARTY.md), LICENSE and NOTICE.
