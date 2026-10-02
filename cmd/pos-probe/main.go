// pos-probe validates trained POS inference and feature hashing against static fixtures.
// It accepts pre-tokenized feature IDs; use g2p-probe for the complete frontend.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/androiddrew/go-g2p/internal/murmur"
	"github.com/androiddrew/ortenv"
	ort "github.com/yalue/onnxruntime_go"
)

type fixture struct {
	ID       string
	Features [][]uint64
	Buckets  [][][]int64
	Scores   [][]float32
	Tags     []string
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func readJSON(path string, value any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, value)
}

func run() error {
	library := flag.String("ort", "", "ORT shared library")
	models := flag.String("models", "models", "directory holding pos.json and pos.onnx")
	fixtures := flag.String("fixtures", "testdata/pos.json", "POS fixture JSON")
	flag.Parse()
	var config struct {
		Labels      []string
		Seeds, Rows []uint32
	}
	if err := readJSON(filepath.Join(*models, "pos.json"), &config); err != nil {
		return err
	}
	if len(config.Labels) != 50 || len(config.Seeds) != 6 || len(config.Rows) != 6 {
		return errors.New("unsupported POS configuration")
	}
	var cases []fixture
	if err := readJSON(*fixtures, &cases); err != nil {
		return err
	}
	lease, err := ortenv.Acquire(*library)
	if err != nil {
		return err
	}
	defer lease.Close()
	options, err := ort.NewSessionOptions()
	if err != nil {
		return err
	}
	defer options.Destroy()
	if err := options.SetIntraOpNumThreads(1); err != nil {
		return err
	}
	if err := options.SetInterOpNumThreads(1); err != nil {
		return err
	}
	start := time.Now()
	session, err := ort.NewDynamicAdvancedSession(filepath.Join(*models, "pos.onnx"), []string{"buckets"}, []string{"vectors", "scores"}, options)
	if err != nil {
		return err
	}
	defer session.Destroy()
	loadNS := time.Since(start).Nanoseconds()
	var maxError float64
	var count int
	var timings []int64
	for _, c := range cases {
		start := time.Now()
		delta, err := check(session, config.Labels, config.Seeds, config.Rows, c)
		if err != nil {
			return fmt.Errorf("case %s: %w", c.ID, err)
		}
		timings = append(timings, time.Since(start).Nanoseconds())
		maxError = math.Max(maxError, delta)
		count += len(c.Tags)
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"cases": len(cases), "tokens": count, "tag_mismatches": 0, "bucket_mismatches": 0, "max_score_error": maxError, "load_ns": loadNS, "case_ns": timings})
}

func check(session *ort.DynamicAdvancedSession, labels []string, seeds, rows []uint32, c fixture) (float64, error) {
	if len(c.Features) == 0 || len(c.Features) != len(c.Tags) || len(c.Buckets) != len(c.Tags) || len(c.Scores) != len(c.Tags) {
		return 0, errors.New("invalid fixture dimensions")
	}
	var flat []int64
	for i, features := range c.Features {
		if len(features) != 6 || len(c.Buckets[i]) != 6 || len(c.Scores[i]) != len(labels) {
			return 0, errors.New("invalid feature dimensions")
		}
		for j, feature := range features {
			if rows[j] == 0 || len(c.Buckets[i][j]) != 4 {
				return 0, errors.New("invalid bucket dimensions")
			}
			for k, hash := range murmur.Hash(feature, seeds[j]) {
				bucket := int64(hash % rows[j])
				if bucket != c.Buckets[i][j][k] {
					return 0, fmt.Errorf("hash mismatch token %d attr %d", i, j)
				}
				flat = append(flat, bucket)
			}
		}
	}
	input, err := ort.NewTensor(ort.NewShape(int64(len(c.Features)), 6, 4), flat)
	if err != nil {
		return 0, err
	}
	defer input.Destroy()
	outputs := []ort.Value{nil, nil}
	defer func() {
		for _, output := range outputs {
			if output != nil {
				_ = output.Destroy()
			}
		}
	}()
	if err = session.Run([]ort.Value{input}, outputs); err != nil {
		return 0, err
	}
	scores, ok := outputs[1].(*ort.Tensor[float32])
	if !ok {
		return 0, errors.New("POS scores must be float32")
	}
	data := scores.GetData()
	if len(data) != len(c.Tags)*len(labels) {
		return 0, errors.New("invalid score shape")
	}
	var delta float64
	for i := range c.Tags {
		best := 0
		for j := range labels {
			value := data[i*len(labels)+j]
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return 0, errors.New("nonfinite scores")
			}
			delta = math.Max(delta, math.Abs(float64(value-c.Scores[i][j])))
			if value > data[i*len(labels)+best] {
				best = j
			}
		}
		if labels[best] != c.Tags[i] {
			return delta, fmt.Errorf("tag mismatch at %d", i)
		}
	}
	if delta > 1e-4 {
		return delta, fmt.Errorf("score error %g exceeds tolerance", delta)
	}
	return delta, nil
}
