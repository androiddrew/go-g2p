// Package espeak is a fallback that runs an external eSpeak-ng executable.
//
// This package contains no eSpeak code and never links eSpeak: it starts the
// espeak-ng program as a separate process. eSpeak-ng itself is GPL-3.0-or-later,
// so whoever distributes the executable alongside an application takes on its
// obligations. Programs that do not import this package depend on no GPL code.
package espeak

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/androiddrew/go-g2p/fallback"
)

// Name identifies this backend in G2P results.
const Name = "espeak"

// Config selects the eSpeak-ng executable.
type Config struct {
	// Executable is a path or PATH name; empty selects "espeak-ng".
	Executable string
}

// Engine is an eSpeak-ng fallback.Engine. Create it with New.
type Engine struct {
	mu         sync.Mutex
	executable string
	closed     bool
}

var _ fallback.Engine = (*Engine)(nil)

// New locates the executable. It does not start a process until Pronounce.
func New(config Config) (*Engine, error) {
	name := config.Executable
	if name == "" {
		name = "espeak-ng"
	}
	path, err := exec.LookPath(name)
	if err != nil {
		return nil, fmt.Errorf("espeak build requires an eSpeak-ng executable; install espeak-ng or set Executable: %w", err)
	}
	return &Engine{executable: path}, nil
}

// Name returns "espeak".
func (e *Engine) Name() string { return Name }

// Rating returns 2, Misaki's rating for eSpeak fallback output.
func (e *Engine) Rating() int { return 2 }

// Close marks the engine closed. It is idempotent.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.closed = true
	return nil
}

// Punctuation must survive the executable, which otherwise discards it.
var punctuation = regexp.MustCompile(`([\s]*[;:,.!?¡¿—…"«»“”(){}\[\]]+[\s]*)+`)
var syllabic = regexp.MustCompile(`(\S)̩`)
var languageFlag = regexp.MustCompile(`\([a-z-]+\)`)

// Pronounce runs espeak-ng for each punctuation-delimited part of text and
// converts its IPA output to Misaki phonemes.
func (e *Engine) Pronounce(ctx context.Context, dialect, text string) (fallback.Result, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return fallback.Result{}, errors.New("espeak engine is closed")
	}
	if err := fallback.CheckDialect(dialect); err != nil {
		return fallback.Result{}, err
	}
	if err := ctx.Err(); err != nil {
		return fallback.Result{}, err
	}
	if !utf8.ValidString(text) || strings.ContainsRune(text, 0) {
		return fallback.Result{}, errors.New("text must be valid UTF-8 without NUL")
	}
	var result strings.Builder
	phonemize := func(part string) error {
		if strings.TrimSpace(part) == "" {
			return nil
		}
		cmd := exec.CommandContext(ctx, e.executable, "-q", "-b", "1", "-v", "en-"+dialect, "--ipa=1", "--tie=^", "--stdin")
		// 1.51's CLI drops the final input byte unless stdin ends in a newline.
		cmd.Stdin = strings.NewReader(part + "\n")
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		if err != nil {
			return fmt.Errorf("eSpeak-ng failed (check executable/voice/data): %w: %s", err, stderr.String())
		}
		// Strip eSpeak's (lang) switch markers and collapse whitespace, as Misaki does.
		ps := languageFlag.ReplaceAllString(string(out), "")
		result.WriteString(strings.Join(strings.Fields(ps), " "))
		return nil
	}
	last := 0
	for _, index := range punctuation.FindAllStringIndex(text, -1) {
		if err := phonemize(text[last:index[0]]); err != nil {
			return fallback.Result{}, err
		}
		result.WriteString(text[index[0]:index[1]])
		last = index[1]
	}
	if err := phonemize(text[last:]); err != nil {
		return fallback.Result{}, err
	}
	return fallback.Result{Phonemes: convertESpeak(strings.TrimSpace(result.String()), dialect == "gb")}, nil
}

// Adapted from hexgrad/misaki EspeakFallback, Apache-2.0; see THIRD_PARTY.md.
func convertESpeak(ps string, british bool) string {
	for _, pair := range [][2]string{
		{"ʔˌn̩", "ʔn"}, {"ʔn̩", "ʔn"}, {"a^ɪ", "I"}, {"a^ʊ", "W"},
		{"d^ʒ", "ʤ"}, {"e^ɪ", "A"}, {"t^ʃ", "ʧ"}, {"ɔ^ɪ", "Y"}, {"ə^l", "ᵊl"},
		{"ʲo", "jo"}, {"ʲə", "jə"}, {"e", "A"}, {"ʲ", ""}, {"ɚ", "əɹ"},
		{"r", "ɹ"}, {"x", "k"}, {"ç", "k"}, {"ɐ", "ə"}, {"ɬ", "l"}, {"̃", ""},
	} {
		ps = strings.ReplaceAll(ps, pair[0], pair[1])
	}
	ps = syllabic.ReplaceAllString(ps, "ᵊ${1}")
	ps = strings.ReplaceAll(ps, "̩", "")
	if british {
		ps = strings.NewReplacer("e^ə", "ɛː", "iə", "ɪə", "ə^ʊ", "Q").Replace(ps)
	} else {
		ps = strings.NewReplacer("o^ʊ", "O", "ɜːɹ", "ɜɹ", "ɜː", "ɜɹ", "ɪə", "iə").Replace(ps)
		ps = strings.ReplaceAll(ps, "ː", "")
	}
	return strings.NewReplacer("o", "ɔ", "ɾ", "T", "ʔ", "t", "^", "").Replace(ps)
}
