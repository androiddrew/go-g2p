package espeak

import (
	"context"
	"os/exec"
	"strings"
	"testing"
)

func TestMissingESpeak(t *testing.T) {
	if _, err := New(Config{Executable: "/nonexistent/espeak-ng"}); err == nil || !strings.Contains(err.Error(), "install espeak-ng") {
		t.Fatalf("expected setup error, got %v", err)
	}
}

func TestESpeakProcess(t *testing.T) {
	if _, err := exec.LookPath("espeak-ng"); err != nil {
		t.Skip("install espeak-ng for real-process checks")
	}
	engine, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	// Real engine checks protect stdin framing, UTF-8, ties, language and punctuation.
	for _, c := range []struct{ dialect, text, want string }{
		{"us", "hello", "həlˈO"}, {"gb", "hello", "həlˈQ"},
		{"us", "Hello, world!", "həlˈO, wˈɜɹld!"},
		{"gb", "(quuxling?)", "(kwˈʌkslɪŋ?)"},
	} {
		got, err := engine.Pronounce(context.Background(), c.dialect, c.text)
		if err != nil || got.Phonemes != c.want {
			t.Errorf("%s %q = %+v, %v; want %s", c.dialect, c.text, got, err, c.want)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := engine.Pronounce(ctx, "us", "hello"); err != context.Canceled {
		t.Fatalf("cancellation: %v", err)
	}
	if _, err := engine.Pronounce(context.Background(), "de", "hello"); err == nil {
		t.Fatal("unsupported dialect accepted")
	}
	if err := engine.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := engine.Pronounce(context.Background(), "us", "hello"); err == nil {
		t.Fatal("use after Close accepted")
	}
}
