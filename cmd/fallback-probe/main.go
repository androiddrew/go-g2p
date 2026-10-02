// fallback-probe runs one fallback backend directly, reading JSONL from stdin.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/androiddrew/go-g2p/fallback"
	"github.com/androiddrew/go-g2p/fallback/espeak"
	"github.com/androiddrew/go-g2p/fallback/neural"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	var neuralConfig neural.Config
	var espeakConfig espeak.Config
	backend := flag.String("fallback", neural.Name, "fallback backend: neural or espeak")
	flag.StringVar(&neuralConfig.ModelDir, "models", "", "neural model directory (default: embedded)")
	flag.StringVar(&neuralConfig.ORTLibrary, "ort", "", "ONNX Runtime library path or OS-loader name (neural)")
	flag.StringVar(&espeakConfig.Executable, "espeak", "", "eSpeak-ng executable (espeak)")
	flag.Parse()
	start := time.Now()
	var engine fallback.Engine
	var err error
	switch *backend {
	case neural.Name:
		engine, err = neural.New(neuralConfig)
	case espeak.Name:
		engine, err = espeak.New(espeakConfig)
	default:
		err = fmt.Errorf("unknown fallback %q: choose neural or espeak", *backend)
	}
	if err != nil {
		return err
	}
	defer engine.Close()
	encoder := json.NewEncoder(os.Stdout)
	if err = encoder.Encode(map[string]any{"backend": engine.Name(), "load_ns": time.Since(start).Nanoseconds()}); err != nil {
		return err
	}
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	for scanner.Scan() {
		var request struct {
			Dialect string `json:"dialect"`
			Text    string `json:"text"`
		}
		if err = json.Unmarshal(scanner.Bytes(), &request); err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		start = time.Now()
		result, callErr := engine.Pronounce(ctx, request.Dialect, request.Text)
		elapsed := time.Since(start).Nanoseconds()
		cancel()
		response := map[string]any{"backend": engine.Name(), "dialect": request.Dialect, "text": request.Text, "result": result, "elapsed_ns": elapsed}
		if callErr != nil {
			response["error"] = callErr.Error()
		}
		if err = encoder.Encode(response); err != nil {
			return err
		}
	}
	return scanner.Err()
}
