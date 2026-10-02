// Package ortruntime decides who owns the process-global ONNX Runtime environment.
package ortruntime

import (
	"errors"

	"github.com/androiddrew/ortenv"
	ort "github.com/yalue/onnxruntime_go"
)

// ErrNotInitialized is returned when the caller owns the environment but has
// not initialized it.
var ErrNotInitialized = errors.New("ONNX Runtime is not initialized: set ORTLibrary, or initialize the environment before calling New")

// Acquire takes an ortenv lease for a non-empty library selector. An empty
// selector means the caller owns the environment: it must already be
// initialized, and the returned nil lease is safe to Close.
func Acquire(library string) (*ortenv.Lease, error) {
	if library != "" {
		return ortenv.Acquire(library)
	}
	if !ort.IsInitialized() {
		return nil, ErrNotInitialized
	}
	return nil, nil
}
