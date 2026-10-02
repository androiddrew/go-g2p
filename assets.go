package g2p

import (
	"embed"
	"io/fs"
	"os"
)

// bundled holds the default frontend data (data/) and POS model (models/). See
// ASSETS.md for provenance.
//
//go:embed data/*.json models/*.json models/*.onnx
var bundled embed.FS

// assetFS returns dir from disk when set, otherwise the embedded sub directory.
func assetFS(dir, embedded string) (fs.FS, error) {
	if dir != "" {
		return os.DirFS(dir), nil
	}
	return fs.Sub(bundled, embedded)
}
