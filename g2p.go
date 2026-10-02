// Package g2p is a Go port of the Misaki US/UK English grapheme-to-phoneme
// frontend, including trained spaCy POS inference. Words missing from the
// dictionaries go to an optional, caller-supplied fallback.Engine.
package g2p

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/androiddrew/go-g2p/fallback"
	"github.com/androiddrew/go-g2p/internal/ortruntime"
	"github.com/androiddrew/ortenv"
)

// ErrNotInitialized is returned by New when ORTLibrary is empty and the caller
// has not initialized the ONNX Runtime environment.
var ErrNotInitialized = ortruntime.ErrNotInitialized

// Dialect selects US or UK English.
type Dialect string

// Supported dialects.
const (
	US Dialect = "us"
	GB Dialect = "gb"
)

// Config locates the frontend assets and selects the ONNX Runtime owner.
type Config struct {
	// DataDir overrides the embedded tokenizer.json, unicode.json and four Misaki
	// dictionary files with a directory holding all six. Usually empty.
	DataDir string
	// ModelDir overrides the embedded pos.json and pos.onnx. Usually empty.
	ModelDir string
	// ORTLibrary is an ONNX Runtime library path or OS-loader name. When set, the
	// engine shares the environment through an ortenv lease. When empty, the
	// caller owns the environment and must initialize it before New and destroy
	// it only after Close.
	ORTLibrary string
	// Threads sets the POS session's intra-op threads. Zero selects 1.
	Threads int
	// Fallback pronounces words missing from the dictionaries, for example a
	// *neural.Engine or *espeak.Engine. The caller owns it and closes it after
	// the G2P engine. When nil, such words are reported as unresolved.
	Fallback fallback.Engine
}

// Request is one text to phonemize.
type Request struct {
	Text    string  `json:"text"`
	Dialect Dialect `json:"dialect"` // required: US or GB
	// AllowTruncated permits fallback output that reached the neural generation
	// limit. Result.Complete stays false and Diagnostics still report it.
	AllowTruncated bool `json:"allow_truncated,omitempty"`
	// Debug fills Result.Debug and FallbackCall.IDs.
	Debug bool `json:"debug,omitempty"`
}

// Token is one word or punctuation token and its pronunciation.
type Token struct {
	Text       string  `json:"text"`
	Tag        string  `json:"tag"` // spaCy fine-grained POS tag
	Whitespace string  `json:"whitespace"`
	Phonemes   *string `json:"phonemes"` // nil when unresolved
	// Source names where the pronunciation came from, e.g. a dictionary,
	// "override" or the fallback's Name.
	Source string `json:"source,omitempty"`
	Rating int    `json:"rating,omitempty"`
	// Start and End are code-point offsets in Result.Preprocessed.
	Start                     int `json:"start"`
	End                       int `json:"end"`
	head, prespace            bool
	alias, currency, numFlags string
	stress                    *float64
	norm                      *string
}

// Diagnostic explains a pronunciation problem, such as an unresolved word.
type Diagnostic struct {
	Code    string `json:"code"` // "unresolved" or "generation_limit"
	Text    string `json:"text"`
	Message string `json:"message"`
}

// FallbackCall records one call to Config.Fallback.
type FallbackCall struct {
	Text         string  `json:"text"`
	Phonemes     string  `json:"phonemes"`
	IDs          []int64 `json:"ids,omitempty"` // only with Request.Debug
	LimitReached bool    `json:"limit_reached,omitempty"`
}

// DebugInfo holds tokenizer and tagger internals, filled when Request.Debug is set.
type DebugInfo struct {
	// TaggedTokens are the tokens after POS tagging, before merging.
	TaggedTokens []Token `json:"tagged_tokens"`
	// Features are the spaCy hash features fed to the POS model.
	Features [][6]uint64 `json:"features"`
}

// Result is the output of Phonemize.
type Result struct {
	// Phonemes is the full Kokoro-compatible phoneme string.
	Phonemes string `json:"phonemes"`
	// Preprocessed is the input after override and normalization handling.
	Preprocessed string `json:"preprocessed"`
	// Backend is the fallback's Name, or empty when no fallback is configured.
	Backend string `json:"backend"`
	// Complete is false when any word is unresolved or generation-limited.
	Complete      bool           `json:"complete"`
	Tokens        []Token        `json:"tokens"`
	FallbackCalls []FallbackCall `json:"fallback_calls"`
	Diagnostics   []Diagnostic   `json:"diagnostics"`
	Debug         *DebugInfo     `json:"debug,omitempty"`
}

// IncompleteError is returned together with a usable Result whose Diagnostics
// describe each unresolved or generation-limited word.
type IncompleteError struct{ Count int }

func (e *IncompleteError) Error() string {
	return fmt.Sprintf("%d unresolved or generation-limited pronunciations; inspect diagnostics and provide overrides", e.Count)
}

// Engine phonemizes text. Requests and Close are serialized.
type Engine struct {
	mu        sync.Mutex
	closed    bool
	tokenizer *tokenizer
	tagger    *tagger
	lexicons  map[Dialect]*lexicon
	fallback  fallback.Engine
	lease     *ortenv.Lease
}

// New loads the dictionaries and POS model for both dialects. Close is mandatory.
func New(c Config) (_ *Engine, err error) {
	e := &Engine{lexicons: map[Dialect]*lexicon{}, fallback: c.Fallback}
	defer func() {
		if err != nil {
			err = errors.Join(err, e.Close())
		}
	}()
	if c.Threads < 0 {
		return nil, errors.New("threads must be nonnegative")
	}
	data, err := assetFS(c.DataDir, "data")
	if err != nil {
		return nil, err
	}
	models, err := assetFS(c.ModelDir, "models")
	if err != nil {
		return nil, err
	}
	e.tokenizer, err = loadTokenizer(data)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	for _, dialect := range []Dialect{US, GB} {
		e.lexicons[dialect], err = loadLexicon(data, string(dialect))
		if err != nil {
			return nil, fmt.Errorf("load %s lexicon: %w", dialect, err)
		}
		e.lexicons[dialect].digits = e.tokenizer.digits
	}
	e.lease, err = ortruntime.Acquire(c.ORTLibrary)
	if err != nil {
		return nil, err
	}
	e.tagger, err = loadTagger(models, c.Threads)
	if err != nil {
		return nil, err
	}
	return e, nil
}

// Close destroys the POS session and releases any ortenv lease. It does not
// close Config.Fallback. Close is idempotent.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	var err error
	if e.tagger != nil {
		err = errors.Join(err, e.tagger.session.Destroy())
	}
	return errors.Join(err, e.lease.Close())
}

// Phonemize converts text to phonemes. When some words cannot be pronounced it
// returns the partial Result together with an *IncompleteError.
func (e *Engine) Phonemize(ctx context.Context, req Request) (Result, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	result := Result{Complete: true, Tokens: []Token{}, FallbackCalls: []FallbackCall{}, Diagnostics: []Diagnostic{}}
	if e.fallback != nil {
		result.Backend = e.fallback.Name()
	}
	if e.closed {
		return result, errors.New("G2P engine is closed")
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	l, ok := e.lexicons[req.Dialect]
	if !ok {
		return result, fmt.Errorf("unsupported dialect %q: choose us or gb", req.Dialect)
	}
	if !utf8.ValidString(req.Text) || strings.ContainsRune(req.Text, 0) {
		return result, errors.New("text must be valid UTF-8 without NUL")
	}
	if utf8.RuneCountInString(req.Text) > 100000 {
		return result, errors.New("text exceeds 100000 code points; split into documents")
	}
	text, overrides := preprocess(req.Text)
	result.Preprocessed = text
	tokens, err := e.tokenizer.tokenize(text)
	if err != nil {
		return result, err
	}
	pos := 0
	for i := range tokens {
		tokens[i].Start = pos
		pos += utf8.RuneCountInString(tokens[i].Text)
		tokens[i].End = pos
		pos += utf8.RuneCountInString(tokens[i].Whitespace)
	}
	features := e.tokenizer.features(tokens)
	if err = e.tagger.tag(tokens, features); err != nil {
		return result, err
	}
	if err = ctx.Err(); err != nil {
		return result, err
	}
	if req.Debug {
		result.Debug = &DebugInfo{TaggedTokens: append([]Token{}, tokens...), Features: features}
	}
	applyFeatures(tokens, overrides)
	folded := []Token{}
	for _, t := range tokens {
		if len(folded) > 0 && !t.head {
			last := len(folded) - 1
			folded[last] = merge([]Token{folded[last], t}, true)
		} else {
			folded = append(folded, t)
		}
	}
	words, err := retokenize(folded)
	if err != nil {
		return result, err
	}
	tc := tokenContext{}
	failures := 0
	fallbackWord := func(t *Token) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if e.fallback == nil {
			result.Diagnostics = append(result.Diagnostics, Diagnostic{"unresolved", t.Text, "not in the dictionaries and no fallback is configured"})
			result.Complete = false
			failures++
			t.Source = "unresolved"
			return nil
		}
		p, err := e.fallback.Pronounce(ctx, string(req.Dialect), t.Text)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		call := FallbackCall{Text: t.Text, Phonemes: p.Phonemes, LimitReached: p.LimitReached}
		if req.Debug {
			call.IDs = p.IDs
		}
		result.FallbackCalls = append(result.FallbackCalls, call)
		if err != nil || p.Phonemes == "" && !all(t.Text, pySpace) {
			message := "fallback returned no pronunciation"
			if err != nil {
				message = err.Error()
			}
			result.Diagnostics = append(result.Diagnostics, Diagnostic{"unresolved", t.Text, message})
			result.Complete = false
			failures++
			t.Source = "unresolved"
			return nil
		}
		t.Phonemes = ptr(p.Phonemes)
		t.Rating = e.fallback.Rating()
		t.Source = e.fallback.Name()
		if p.LimitReached {
			result.Complete = false
			result.Diagnostics = append(result.Diagnostics, Diagnostic{"generation_limit", t.Text, "neural generation reached 21 tokens and may omit sounds; split the span or use a pronunciation override"})
			if !req.AllowTruncated {
				failures++
			}
		}
		return nil
	}
	for i := len(words) - 1; i >= 0; i-- {
		if err = ctx.Err(); err != nil {
			return result, err
		}
		w := words[i]
		if len(w) == 1 {
			t := &w[0]
			if t.Phonemes == nil {
				setPronunciation(t, l.pronounce(*t, tc))
			}
			if t.Phonemes == nil {
				if err = fallbackWord(t); err != nil {
					return result, err
				}
			}
			tc = nextContext(tc, *t)
			continue
		}
		left, right := 0, len(w)
		shouldFallback := false
		for left < right {
			resolved := false
			for _, t := range w[left:right] {
				resolved = resolved || t.alias != "" || t.Phonemes != nil
			}
			t := merge(w[left:right], false)
			p := pronunciation{}
			if !resolved {
				p = l.pronounce(t, tc)
			}
			if p.known {
				setPronunciation(&w[left], p)
				for j := left + 1; j < right; j++ {
					w[j].Phonemes = ptr("")
					w[j].Rating = p.rating
					w[j].Source = p.source
				}
				t.Phonemes = ptr(p.phones)
				tc = nextContext(tc, t)
				right = left
				left = 0
			} else if left+1 < right {
				left++
			} else {
				right--
				t := &w[right]
				if t.Phonemes == nil {
					if all(t.Text, func(r rune) bool { return has(junks, r) }) {
						setPronunciation(t, known("", 3, "separator"))
					} else {
						shouldFallback = true
						break
					}
				}
				left = 0
			}
		}
		if shouldFallback {
			t := merge(w, false)
			if err = fallbackWord(&t); err != nil {
				return result, err
			}
			w[0].Phonemes = t.Phonemes
			w[0].Rating = t.Rating
			w[0].Source = t.Source
			for j := 1; j < len(w); j++ {
				w[j].Phonemes = ptr("")
				w[j].Rating = t.Rating
				w[j].Source = t.Source
			}
		} else {
			resolve(w)
		}
	}
	var b strings.Builder
	for _, w := range words {
		t := w[0]
		if len(w) > 1 {
			t = merge(w, true)
		}
		if t.Phonemes != nil {
			p := strings.NewReplacer("ɾ", "T", "ʔ", "t").Replace(*t.Phonemes)
			t.Phonemes = &p
			b.WriteString(p)
		} else {
			b.WriteString("❓")
		}
		b.WriteString(t.Whitespace)
		result.Tokens = append(result.Tokens, t)
	}
	result.Phonemes = b.String()
	if failures > 0 {
		return result, &IncompleteError{failures}
	}
	return result, nil
}

func setPronunciation(t *Token, p pronunciation) {
	if p.known {
		t.Phonemes = ptr(p.phones)
		t.Rating = p.rating
		t.Source = p.source
	}
}
