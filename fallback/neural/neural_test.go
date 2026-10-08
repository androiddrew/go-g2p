package neural

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/androiddrew/go-g2p/internal/nativetest"
	"github.com/androiddrew/go-g2p/internal/ortruntime"
	"github.com/androiddrew/ortenv"
	ort "github.com/yalue/onnxruntime_go"
)

func TestMissingNeuralDependencies(t *testing.T) {
	if !nativetest.Subprocess(t) {
		return
	}
	if _, err := New(Config{ModelDir: t.TempDir()}); err == nil || !strings.Contains(err.Error(), "us.json") {
		t.Fatalf("expected actionable setup error, got %v", err)
	}
	if _, err := New(Config{}); !errors.Is(err, ortruntime.ErrNotInitialized) {
		t.Fatalf("embedded models without a runtime: %v", err)
	}
	if _, err := New(Config{Threads: -1}); err == nil {
		t.Fatal("negative threads accepted")
	}
}

func TestCallerOwnedEnvironment(t *testing.T) {
	library := os.Getenv("G2P_TEST_ORT")
	if library == "" {
		t.Skip("set G2P_TEST_ORT for real-model lifecycle checks")
	}
	if !nativetest.Subprocess(t) {
		return
	}
	if _, err := New(Config{}); !errors.Is(err, ortruntime.ErrNotInitialized) {
		t.Fatalf("uninitialized caller-owned environment: %v", err)
	}
	if err := ortenv.Init(library); err != nil {
		t.Fatal(err)
	}
	engine, err := New(Config{Threads: 2})
	if err != nil {
		t.Fatal(err)
	}
	if result, err := engine.Pronounce(context.Background(), "us", "hello"); err != nil || result.Phonemes != "hˈɛlO" {
		t.Errorf("caller-owned pronunciation: %+v, %v", result, err)
	}
	if err = engine.Close(); err != nil {
		t.Fatal(err)
	}
	if !ort.IsInitialized() {
		t.Fatal("Close destroyed a caller-owned environment")
	}
}

func TestNeuralLifecycle(t *testing.T) {
	library := os.Getenv("G2P_TEST_ORT")
	if library == "" {
		t.Skip("set G2P_TEST_ORT for real-model lifecycle checks")
	}
	models, err := filepath.Abs("models")
	if err != nil {
		t.Fatal(err)
	}
	config := Config{ModelDir: models, ORTLibrary: library}
	bad := t.TempDir()
	// Valid US sessions load first, then malformed UK decoder fails. All native
	// sessions must be released; the environment is retained for the next New.
	for _, name := range []string{"us.json", "gb.json", "us-encoder.onnx", "us-decoder.onnx", "gb-encoder.onnx"} {
		if err := os.Symlink(filepath.Join(models, name), filepath.Join(bad, name)); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(bad, "gb-decoder.onnx"), []byte("not ONNX"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := New(Config{ModelDir: bad, ORTLibrary: library}); err == nil {
		t.Fatal("malformed model accepted")
	}
	if !ort.IsInitialized() {
		t.Fatal("partial initialization destroyed the environment")
	}
	for range 3 {
		engine, err := New(config)
		if err != nil {
			t.Fatal(err)
		}
		second, err := New(config)
		if err != nil {
			t.Fatal(err)
		}
		if err := second.Close(); err != nil {
			t.Fatal(err)
		}
		if !ort.IsInitialized() {
			t.Fatal("second owner unloaded live engine")
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := engine.Pronounce(ctx, "us", "hello"); err != context.Canceled {
			t.Fatalf("cancellation: %v", err)
		}
		for _, input := range []struct{ dialect, text string }{{"au", "hello"}, {"us", strings.Repeat("a", 63)}, {"us", string([]byte{255})}} {
			if _, err := engine.Pronounce(context.Background(), input.dialect, input.text); err == nil {
				t.Fatalf("invalid input accepted: %+v", input)
			}
		}
		var wg sync.WaitGroup
		for range 4 {
			wg.Go(func() {
				result, err := engine.Pronounce(context.Background(), "us", "hello")
				// Fixed direct-fallback regression value, not the dictionary result.
				if err != nil || result.Phonemes != "hˈɛlO" {
					t.Errorf("concurrent pronunciation: %+v, %v", result, err)
				}
			})
		}
		wg.Wait()
		if err := engine.Close(); err != nil {
			t.Fatal(err)
		}
		if err := engine.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := engine.Pronounce(context.Background(), "us", "hello"); err == nil {
			t.Fatal("use after Close accepted")
		}
		// ortenv v0.1.0 unloaded the runtime here and reloaded it on the next
		// iteration, which crashes CUDA providers (androiddrew/ortenv#2).
		if !ort.IsInitialized() {
			t.Fatal("last Close destroyed the environment")
		}
	}
}
