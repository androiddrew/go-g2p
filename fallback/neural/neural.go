// Package neural is the GPL-free fallback: Misaki's US/UK grapheme-to-phoneme
// encoder/decoder models run through ONNX Runtime.
package neural

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"sync"
	"unicode/utf8"

	"github.com/androiddrew/go-g2p/fallback"
	"github.com/androiddrew/go-g2p/internal/ortruntime"
	"github.com/androiddrew/ortenv"
	ort "github.com/yalue/onnxruntime_go"
)

// bundled holds the default US/UK models. See the go-g2p ASSETS.md for provenance.
//
//go:embed models/*.json models/*.onnx
var bundled embed.FS

// Name identifies this backend in G2P results.
const Name = "neural"

// Config locates the neural models and selects the ONNX Runtime owner.
type Config struct {
	// ModelDir overrides the embedded models with a directory holding us.json,
	// gb.json and each dialect's -encoder.onnx and -decoder.onnx. Usually empty.
	ModelDir string
	// ORTLibrary is an ONNX Runtime library path or OS-loader name. When set, the
	// engine shares the environment through an ortenv lease. When empty, the
	// caller owns the environment and must initialize it before New.
	ORTLibrary string
	// Threads sets intra-op threads per session. Zero selects 1.
	Threads int
}

type neuralConfig struct {
	Graphemes    string `json:"graphemes"`
	Phonemes     string `json:"phonemes"`
	MaxPositions int    `json:"max_positions"`
	MaxLength    int    `json:"max_length"`
}

type network struct {
	config           neuralConfig
	encoder, decoder *ort.DynamicAdvancedSession
}

// Engine is a neural fallback.Engine. Create it with New.
type Engine struct {
	mu     sync.Mutex
	models map[string]*network
	closed bool
	lease  *ortenv.Lease
}

var _ fallback.Engine = (*Engine)(nil)

// New loads the US and UK models. Close is mandatory.
func New(config Config) (*Engine, error) {
	if config.Threads < 0 {
		return nil, errors.New("threads must be nonnegative")
	}
	threads := config.Threads
	if threads == 0 {
		threads = 1
	}
	var models fs.FS = os.DirFS(config.ModelDir)
	if config.ModelDir == "" {
		var err error
		if models, err = fs.Sub(bundled, "models"); err != nil {
			return nil, err
		}
	}
	n := &Engine{models: make(map[string]*network)}
	for _, dialect := range []string{"us", "gb"} {
		data, err := fs.ReadFile(models, dialect+".json")
		if err != nil {
			return nil, fmt.Errorf("load neural %s config (see ASSETS.md): %w", dialect, err)
		}
		m := &network{}
		if err = json.Unmarshal(data, &m.config); err != nil {
			return nil, err
		}
		if m.config.MaxPositions != 64 || m.config.MaxLength != 21 || len([]rune(m.config.Graphemes)) < 4 || len([]rune(m.config.Phonemes)) < 4 {
			return nil, fmt.Errorf("unsupported neural configuration for %s", dialect)
		}
		n.models[dialect] = m
	}
	var err error
	n.lease, err = ortruntime.Acquire(config.ORTLibrary)
	if err != nil {
		return nil, err
	}
	options, err := ort.NewSessionOptions()
	if err != nil {
		_ = n.lease.Close()
		return nil, err
	}
	if err = options.SetIntraOpNumThreads(threads); err == nil {
		err = options.SetInterOpNumThreads(1)
	}
	if err == nil {
		for _, dialect := range []string{"us", "gb"} {
			m := n.models[dialect]
			var encoder, decoder []byte
			if encoder, err = fs.ReadFile(models, dialect+"-encoder.onnx"); err != nil {
				break
			}
			m.encoder, err = ort.NewDynamicAdvancedSessionWithONNXData(encoder, []string{"input_ids"}, []string{"hidden"}, options)
			if err != nil {
				break
			}
			if decoder, err = fs.ReadFile(models, dialect+"-decoder.onnx"); err != nil {
				break
			}
			m.decoder, err = ort.NewDynamicAdvancedSessionWithONNXData(decoder, []string{"decoder_ids", "hidden"}, []string{"logits"}, options)
			if err != nil {
				break
			}
		}
	}
	// Options must be released before unloading the ORT library on error paths.
	err = errors.Join(err, options.Destroy())
	if err != nil {
		err = errors.Join(err, n.destroySessions(), n.lease.Close())
		return nil, fmt.Errorf("load neural ONNX assets (see ASSETS.md): %w", err)
	}
	return n, nil
}

func (n *Engine) destroySessions() error {
	var errs []error
	for _, m := range n.models {
		if m.decoder != nil {
			errs = append(errs, m.decoder.Destroy())
		}
		if m.encoder != nil {
			errs = append(errs, m.encoder.Destroy())
		}
	}
	return errors.Join(errs...)
}

// Name returns "neural".
func (n *Engine) Name() string { return Name }

// Rating returns 1, Misaki's rating for neural fallback output.
func (n *Engine) Rating() int { return 1 }

// Close destroys the sessions and releases any ortenv lease. It is idempotent.
func (n *Engine) Close() error {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.closed {
		return nil
	}
	n.closed = true
	err := n.destroySessions()
	err = errors.Join(err, n.lease.Close())
	return err
}

func run(session *ort.DynamicAdvancedSession, inputs []ort.Value) (ort.Value, error) {
	outputs := []ort.Value{nil}
	err := session.Run(inputs, outputs)
	if err != nil {
		if outputs[0] != nil {
			_ = outputs[0].Destroy()
		}
		return nil, err
	}
	return outputs[0], nil
}

// Pronounce generates phonemes for at most 62 code points of text.
func (n *Engine) Pronounce(ctx context.Context, dialect, text string) (fallback.Result, error) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.closed {
		return fallback.Result{}, errors.New("neural engine is closed")
	}
	if err := fallback.CheckDialect(dialect); err != nil {
		return fallback.Result{}, err
	}
	if err := ctx.Err(); err != nil {
		return fallback.Result{}, err
	}
	if !utf8.ValidString(text) {
		return fallback.Result{}, errors.New("text must be valid UTF-8")
	}
	m := n.models[dialect]
	if utf8.RuneCountInString(text) > m.config.MaxPositions-2 {
		return fallback.Result{}, errors.New("neural fallback supports at most 62 Unicode code points; split/override the unresolved span")
	}
	table := map[rune]int64{}
	for i, c := range []rune(m.config.Graphemes) {
		table[c] = int64(i)
	}
	ids := []int64{1}
	for _, c := range text {
		id, ok := table[c]
		if !ok {
			id = 3
		}
		ids = append(ids, id)
	}
	ids = append(ids, 2)
	input, err := ort.NewTensor(ort.NewShape(1, int64(len(ids))), ids)
	if err != nil {
		return fallback.Result{}, err
	}
	defer input.Destroy()
	hidden, err := run(m.encoder, []ort.Value{input})
	if err != nil {
		return fallback.Result{}, err
	}
	defer hidden.Destroy()
	generated := []int64{1}
	for len(generated) < m.config.MaxLength {
		if err := ctx.Err(); err != nil {
			return fallback.Result{}, err
		}
		target, err := ort.NewTensor(ort.NewShape(1, int64(len(generated))), generated)
		if err != nil {
			return fallback.Result{}, err
		}
		logits, err := run(m.decoder, []ort.Value{target, hidden})
		_ = target.Destroy()
		if err != nil {
			return fallback.Result{}, err
		}
		values, ok := logits.(*ort.Tensor[float32])
		if !ok {
			_ = logits.Destroy()
			return fallback.Result{}, errors.New("decoder output must be float32")
		}
		data := values.GetData()
		if len(data) != 63 {
			_ = logits.Destroy()
			return fallback.Result{}, fmt.Errorf("unexpected decoder vocabulary: %d", len(data))
		}
		best := 0
		for i, value := range data {
			if value > data[best] {
				best = i
			}
		}
		_ = logits.Destroy()
		if len(generated) == m.config.MaxLength-1 {
			best = 2
		}
		generated = append(generated, int64(best))
		if best == 2 {
			break
		}
	}
	phonemes := []rune(m.config.Phonemes)
	var result []rune
	for _, id := range generated {
		if id > 3 && id < int64(len(phonemes)) {
			result = append(result, phonemes[id])
		}
	}
	// Like Misaki's fallback, return raw phones. The G2P engine maps flaps and
	// glottal stops (ɾ→T, ʔ→t) afterwards, so this output matches Misaki's.
	return fallback.Result{Phonemes: string(result), IDs: generated, LimitReached: len(generated) == m.config.MaxLength}, nil
}
