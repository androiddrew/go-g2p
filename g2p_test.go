package g2p

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/androiddrew/go-g2p/fallback"
	"github.com/androiddrew/go-g2p/fallback/espeak"
	"github.com/androiddrew/go-g2p/fallback/neural"
	"github.com/androiddrew/ortenv"
	ort "github.com/yalue/onnxruntime_go"
)

func testConfig(t *testing.T) Config {
	t.Helper()
	c := Config{ORTLibrary: os.Getenv("G2P_TEST_ORT")}
	if c.ORTLibrary == "" {
		t.Skip("set G2P_TEST_ORT to an ONNX Runtime library path or loader name")
	}
	return c
}

// testBackends returns each fallback that can run here: neural always, eSpeak
// only when espeak-ng is installed. The test closes them after the engines.
func testBackends(t *testing.T, c Config) []fallback.Engine {
	t.Helper()
	n, err := neural.New(neural.Config{ORTLibrary: c.ORTLibrary})
	if err != nil {
		t.Fatal(err)
	}
	backends := []fallback.Engine{n}
	if _, err := exec.LookPath("espeak-ng"); err == nil {
		e, err := espeak.New(espeak.Config{})
		if err != nil {
			t.Fatal(err)
		}
		backends = append(backends, e)
	} else {
		t.Log("espeak-ng not installed; skipping eSpeak backend")
	}
	t.Cleanup(func() {
		for _, b := range backends {
			if err := b.Close(); err != nil {
				t.Error(err)
			}
		}
	})
	return backends
}

func TestFrontendLifecycle(t *testing.T) {
	base := testConfig(t)
	for _, backend := range testBackends(t, base) {
		t.Run(backend.Name(), func(t *testing.T) {
			c := base
			c.Fallback = backend
			for range 2 {
				e, err := New(c)
				if err != nil {
					t.Fatal(err)
				}
				other, err := New(c)
				if err != nil {
					t.Fatal(err)
				}
				if err = other.Close(); err != nil {
					t.Fatal(err)
				}
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				if _, err = e.Phonemize(ctx, Request{Text: "hello", Dialect: US}); !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
				for _, req := range []Request{{Text: "hello", Dialect: "au"}, {Text: "\xff", Dialect: US}, {Text: "\x00", Dialect: US}, {Text: strings.Repeat("x", 100001), Dialect: US}} {
					if _, err = e.Phonemize(context.Background(), req); err == nil {
						t.Fatalf("accepted invalid request: %.30q", req.Text)
					}
				}
				var wg sync.WaitGroup
				for range 4 {
					wg.Go(func() {
						r, err := e.Phonemize(context.Background(), Request{Text: "Hello world!", Dialect: US})
						if err != nil || r.Phonemes != "həlˈO wˈɜɹld!" || !r.Complete || r.Backend != backend.Name() {
							t.Errorf("%+v %v", r, err)
						}
					})
				}
				wg.Wait()
				for _, text := range []string{"", " \t\n"} {
					r, err := e.Phonemize(context.Background(), Request{Text: text, Dialect: GB})
					if err != nil || r.Phonemes != "" {
						t.Fatalf("empty: %+v %v", r, err)
					}
				}
				// A complete explicit override bypasses both dictionaries and fallback.
				r, err := e.Phonemize(context.Background(), Request{Text: "[very obscure name](/həlˈO/)", Dialect: US})
				if err != nil || r.Phonemes != "həlˈO" || len(r.FallbackCalls) != 0 {
					t.Fatalf("override %+v %v", r, err)
				}
				r, err = e.Phonemize(context.Background(), Request{Text: "Hello, outofdictionary world!", Dialect: US})
				if err != nil || len(r.FallbackCalls) != 1 || r.Debug != nil || r.FallbackCalls[0].IDs != nil {
					t.Fatalf("fallback call %+v %v", r, err)
				}
				if source := r.Tokens[2].Source; source != backend.Name() {
					t.Fatalf("fallback token source %q", source)
				}
				if backend.Name() == neural.Name {
					r, err = e.Phonemize(context.Background(), Request{Text: strings.Repeat("q", 63), Dialect: US})
					var incomplete *IncompleteError
					if !errors.As(err, &incomplete) || r.Complete || !strings.Contains(r.Phonemes, "❓") {
						t.Fatalf("long unresolved %+v %v", r, err)
					}
					r, err = e.Phonemize(context.Background(), Request{Text: "https://example.com/path?q=1", Dialect: US})
					if !errors.As(err, &incomplete) || r.Complete || len(r.Diagnostics) == 0 {
						t.Fatalf("generation limit %+v %v", r, err)
					}
				}
				if err = e.Close(); err != nil {
					t.Fatal(err)
				}
				if err = e.Close(); err != nil {
					t.Fatal(err)
				}
				if _, err = e.Phonemize(context.Background(), Request{Text: "hello", Dialect: US}); err == nil {
					t.Fatal("use after Close")
				}
				// The caller-owned fallback must survive the engine's Close.
				if _, err = backend.Pronounce(context.Background(), "us", "hello"); err != nil {
					t.Fatalf("engine closed the injected fallback: %v", err)
				}
			}
		})
	}
}

func TestOverrideDirectories(t *testing.T) {
	c := testConfig(t)
	c.DataDir, c.ModelDir = "data", "models"
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if r, err := e.Phonemize(context.Background(), Request{Text: "Hello world!", Dialect: GB}); err != nil || r.Phonemes != "həlˈQ wˈɜːld!" {
		t.Fatalf("on-disk assets: %+v %v", r, err)
	}
}

func TestNoFallback(t *testing.T) {
	e, err := New(testConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	r, err := e.Phonemize(context.Background(), Request{Text: "Hello, outofdictionary world!", Dialect: US, Debug: true})
	var incomplete *IncompleteError
	if !errors.As(err, &incomplete) || r.Complete || r.Backend != "" || len(r.FallbackCalls) != 0 || len(r.Diagnostics) != 1 || r.Diagnostics[0].Code != "unresolved" {
		t.Fatalf("unresolved without fallback: %+v %v", r, err)
	}
	if r.Debug == nil || len(r.Debug.TaggedTokens) == 0 || len(r.Debug.Features) != len(r.Debug.TaggedTokens) {
		t.Fatalf("debug info: %+v", r.Debug)
	}
	if _, err = New(Config{DataDir: "x", ModelDir: "x", Threads: -1}); err == nil {
		t.Fatal("negative threads accepted")
	}
}

func TestCallerOwnedEnvironment(t *testing.T) {
	c := testConfig(t)
	library := c.ORTLibrary
	c.ORTLibrary = ""
	if _, err := New(c); !errors.Is(err, ErrNotInitialized) {
		t.Fatalf("uninitialized caller-owned environment: %v", err)
	}
	lease, err := ortenv.Acquire(library)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := lease.Close(); err != nil {
			t.Error(err)
		}
	}()
	n, err := neural.New(neural.Config{})
	if err != nil {
		t.Fatal(err)
	}
	defer n.Close()
	c.Fallback, c.Threads = n, 2
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	if r, err := e.Phonemize(context.Background(), Request{Text: "Hello, outofdictionary world!", Dialect: US}); err != nil || !r.Complete {
		t.Fatalf("caller-owned phonemize: %+v %v", r, err)
	}
	if err = e.Close(); err != nil {
		t.Fatal(err)
	}
	if !ort.IsInitialized() {
		t.Fatal("Close destroyed a caller-owned environment")
	}
}

func TestMalformedFrontend(t *testing.T) {
	c := testConfig(t)
	source, err := filepath.Abs("data")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"tokenizer.json", "unicode.json", "us_gold.json"} {
		dir := t.TempDir()
		for _, file := range []string{"tokenizer.json", "unicode.json", "us_gold.json", "us_silver.json", "gb_gold.json", "gb_silver.json"} {
			if file == name {
				if err := os.WriteFile(filepath.Join(dir, file), []byte(`{}`), 0600); err != nil {
					t.Fatal(err)
				}
			} else if err := os.Symlink(filepath.Join(source, file), filepath.Join(dir, file)); err != nil {
				t.Fatal(err)
			}
		}
		bad := c
		bad.DataDir = dir
		if _, err := New(bad); err == nil {
			t.Fatalf("accepted malformed %s", name)
		}
		if ort.IsInitialized() {
			t.Fatal("leaked environment")
		}
	}
	bad := c
	bad.ModelDir = t.TempDir()
	if err := os.Symlink(filepath.Join(source, "..", "models", "pos.json"), filepath.Join(bad.ModelDir, "pos.json")); err != nil {
		t.Fatal(err)
	}
	if _, err := New(bad); err == nil {
		t.Fatal("accepted missing POS model")
	}
	if ort.IsInitialized() {
		t.Fatal("leaked environment after failed POS load")
	}
}

func TestTokenizerFixtures(t *testing.T) {
	c := testConfig(t)
	path := os.Getenv("G2P_TEST_TOKENIZER")
	if path == "" {
		path = "testdata/tokenizer.json"
	}
	e, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	var fixtures []struct {
		Text     string
		Tokens   []Token
		Features [][6]uint64
	}
	if err = readJSON(path, &fixtures); err != nil {
		t.Fatal(err)
	}
	failures := 0
	tokens := 0
	for _, f := range fixtures {
		got, err := e.tokenizer.tokenize(f.Text)
		if err != nil {
			t.Fatal(err)
		}
		features := e.tokenizer.features(got)
		if err = e.tagger.tag(got, features); err != nil {
			t.Fatal(err)
		}
		match := len(got) == len(f.Tokens) && reflect.DeepEqual(features, f.Features)
		for i := 0; i < min(len(got), len(f.Tokens)); i++ {
			match = match && got[i].Text == f.Tokens[i].Text && got[i].Whitespace == f.Tokens[i].Whitespace && got[i].Tag == f.Tokens[i].Tag
		}
		if !match {
			failures++
			if failures <= 12 {
				a, _ := json.Marshal(got)
				b, _ := json.Marshal(f.Tokens)
				t.Logf("%q\ngot %s\nwant %s\nfeatures match %v", f.Text, a, b, reflect.DeepEqual(features, f.Features))
			}
		}
		tokens += len(f.Tokens)
	}
	t.Logf("%d cases, %d tokens, %d mismatches", len(fixtures), tokens, failures)
	if failures > 0 {
		t.Fail()
	}
}

func TestPipelineFixtures(t *testing.T) {
	base := testConfig(t)
	var fixtures []struct {
		ID            string         `json:"id"`
		Backend       string         `json:"backend"`
		Request       Request        `json:"request"`
		Preprocessed  string         `json:"preprocessed"`
		Phonemes      string         `json:"phonemes"`
		Tokens        []Token        `json:"tokens"`
		TaggedTokens  []Token        `json:"tagged_tokens"`
		Features      [][6]uint64    `json:"features"`
		FallbackCalls []FallbackCall `json:"fallback_calls"`
	}
	if err := readJSON("testdata/pipeline.json", &fixtures); err != nil {
		t.Fatal(err)
	}
	for _, backend := range testBackends(t, base) {
		c := base
		c.Fallback = backend
		e, err := New(c)
		if err != nil {
			t.Fatal(err)
		}
		defer e.Close()
		ran := 0
		for _, f := range fixtures {
			if f.Backend != backend.Name() {
				continue
			}
			ran++
			t.Run(f.Backend+"/"+f.ID+"/"+string(f.Request.Dialect), func(t *testing.T) {
				f.Request.Debug = true
				got, err := e.Phonemize(context.Background(), f.Request)
				if err != nil {
					t.Fatal(err)
				}
				if got.Preprocessed != f.Preprocessed || got.Phonemes != f.Phonemes || !reflect.DeepEqual(got.Debug.Features, f.Features) {
					t.Fatalf("pipeline drift: %q != %q", got.Phonemes, f.Phonemes)
				}
				for _, pair := range [][2][]Token{{got.Tokens, f.Tokens}, {got.Debug.TaggedTokens, f.TaggedTokens}} {
					if len(pair[0]) != len(pair[1]) {
						t.Fatal("token count drift")
					}
					for i, a := range pair[0] {
						b := pair[1][i]
						if a.Text != b.Text || a.Tag != b.Tag || a.Whitespace != b.Whitespace || !reflect.DeepEqual(a.Phonemes, b.Phonemes) {
							t.Fatalf("token %d drift: %+v != %+v", i, a, b)
						}
					}
				}
				if len(got.FallbackCalls) != len(f.FallbackCalls) {
					t.Fatal("fallback count drift")
				}
				for i, a := range got.FallbackCalls {
					b := f.FallbackCalls[i]
					if a.Text != b.Text || a.Phonemes != b.Phonemes {
						t.Fatal("fallback drift")
					}
				}
			})
		}
		if ran == 0 || ran*2 != len(fixtures) {
			t.Fatalf("invalid %s fixture coverage: %d of %d", backend.Name(), ran, len(fixtures))
		}
		t.Logf("validated %d complete %s pipeline fixtures", ran, backend.Name())
	}
}
