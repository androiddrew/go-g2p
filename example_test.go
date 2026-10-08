package g2p_test

import (
	"context"
	"errors"
	"fmt"
	"log"

	g2p "github.com/androiddrew/go-g2p"
	"github.com/androiddrew/go-g2p/fallback/neural"
	ort "github.com/yalue/onnxruntime_go"
)

// The usual setup: the engine and its neural fallback share ONNX Runtime through
// ortenv. Dictionaries and models are embedded, so no paths are needed.
func Example() {
	const library = "libonnxruntime.so" // OS loader name or a path to the library

	fb, err := neural.New(neural.Config{ORTLibrary: library})
	if err != nil {
		log.Fatal(err)
	}
	defer fb.Close() // the engine does not close its fallback

	engine, err := g2p.New(g2p.Config{ORTLibrary: library, Fallback: fb})
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()

	result, err := engine.Phonemize(context.Background(), g2p.Request{Text: "Hello, world!", Dialect: g2p.US})
	var incomplete *g2p.IncompleteError
	if errors.As(err, &incomplete) {
		for _, d := range result.Diagnostics {
			fmt.Printf("%s: %q: %s\n", d.Code, d.Text, d.Message)
		}
	} else if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Phonemes) // həlˈO, wˈɜɹld!
}

// An application that already initializes ONNX Runtime itself leaves
// ORTLibrary empty. The engine then uses that environment and never destroys it.
func Example_callerOwnedEnvironment() {
	ort.SetSharedLibraryPath("libonnxruntime.so")
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}
	defer ort.DestroyEnvironment() // runs after engine.Close

	engine, err := g2p.New(g2p.Config{}) // no fallback: unknown words are reported
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()

	result, err := engine.Phonemize(context.Background(), g2p.Request{Text: "Good morning.", Dialect: g2p.GB})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Phonemes)
}
