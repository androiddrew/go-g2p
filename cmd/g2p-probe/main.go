// g2p-probe is a JSON-lines validation driver, not the application CLI.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	g2p "github.com/androiddrew/go-g2p"
	"github.com/androiddrew/go-g2p/fallback/espeak"
	"github.com/androiddrew/go-g2p/fallback/neural"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() (err error) {
	c := g2p.Config{}
	flag.StringVar(&c.DataDir, "data", "", "dictionary/tokenizer directory (default: embedded)")
	flag.StringVar(&c.ModelDir, "models", "", "POS model directory (default: embedded)")
	neuralModels := flag.String("neural-models", "", "neural fallback model directory (default: embedded)")
	flag.StringVar(&c.ORTLibrary, "ort", "", "ORT shared library")
	backend := flag.String("fallback", neural.Name, "fallback backend: neural, espeak or none")
	executable := flag.String("espeak", "", "eSpeak-ng executable")
	flag.Parse()
	switch *backend {
	case neural.Name:
		n, err := neural.New(neural.Config{ModelDir: *neuralModels, ORTLibrary: c.ORTLibrary})
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, n.Close()) }()
		c.Fallback = n
	case espeak.Name:
		s, err := espeak.New(espeak.Config{Executable: *executable})
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, s.Close()) }()
		c.Fallback = s
	case "none":
	default:
		return fmt.Errorf("unknown fallback %q: choose neural, espeak or none", *backend)
	}
	e, err := g2p.New(c)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, e.Close()) }()
	s := bufio.NewScanner(os.Stdin)
	s.Buffer(make([]byte, 4096), 2000000)
	out := json.NewEncoder(os.Stdout)
	for s.Scan() {
		var req g2p.Request
		if err = json.Unmarshal(s.Bytes(), &req); err != nil {
			return err
		}
		result, e := e.Phonemize(context.Background(), req)
		message := ""
		if e != nil {
			message = e.Error()
		}
		if err = out.Encode(struct {
			g2p.Result
			Error string `json:"error,omitempty"`
		}{result, message}); err != nil {
			return err
		}
	}
	return s.Err()
}
