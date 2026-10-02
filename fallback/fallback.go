// Package fallback defines the interface for pronouncing words that the G2P
// dictionaries cannot resolve. Implementations live in the neural and espeak
// subpackages; applications construct one and pass it to g2p.Config.Fallback.
package fallback

import (
	"context"
	"fmt"
)

// Result is one fallback pronunciation.
type Result struct {
	// Phonemes are the raw phones produced by the backend.
	Phonemes string `json:"phonemes"`
	// IDs are backend-specific generated token IDs (neural only), for diagnostics.
	IDs []int64 `json:"ids,omitempty"`
	// LimitReached reports that generation stopped at the backend's length limit,
	// so the pronunciation may be missing sounds.
	LimitReached bool `json:"limit_reached,omitempty"`
}

// Engine pronounces a single unresolved span. Implementations must serialize
// calls, reject calls after Close, and make Close idempotent.
type Engine interface {
	// Pronounce returns phonemes for text in the given dialect ("us" or "gb").
	Pronounce(ctx context.Context, dialect, text string) (Result, error)
	// Name identifies the backend in results and diagnostics, e.g. "neural".
	Name() string
	// Rating is the confidence rating attached to tokens this backend resolves.
	// Misaki rates neural output 1 and eSpeak output 2.
	Rating() int
	// Close releases the engine's native resources.
	Close() error
}

// CheckDialect reports whether dialect is one of the supported "us" or "gb".
// It is exported for Engine implementations.
func CheckDialect(dialect string) error {
	if dialect != "us" && dialect != "gb" {
		return fmt.Errorf("unsupported dialect %q: choose us or gb", dialect)
	}
	return nil
}
