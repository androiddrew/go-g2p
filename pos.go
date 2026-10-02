package g2p

import (
	"errors"
	"fmt"
	"io/fs"
	"math"

	"github.com/androiddrew/go-g2p/internal/murmur"
	ort "github.com/yalue/onnxruntime_go"
)

type tagger struct {
	Labels      []string
	Seeds, Rows []uint32
	session     *ort.DynamicAdvancedSession
}

func loadTagger(fsys fs.FS, threads int) (result *tagger, err error) {
	t := new(tagger)
	if err = readFSJSON(fsys, "pos.json", t); err != nil {
		return nil, err
	}
	if len(t.Labels) != 50 || len(t.Seeds) != 6 || len(t.Rows) != 6 {
		return nil, errors.New("unsupported POS configuration")
	}
	for i, rows := range []uint32{5000, 1000, 2500, 2500, 50, 50} {
		if t.Rows[i] != rows || t.Seeds[i] != uint32(i+8) {
			return nil, errors.New("unsupported POS hash configuration")
		}
	}
	model, err := fs.ReadFile(fsys, "pos.onnx")
	if err != nil {
		return nil, err
	}
	o, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer func() {
		err = errors.Join(err, o.Destroy())
		if err != nil && t.session != nil {
			err = errors.Join(err, t.session.Destroy())
		}
		if err != nil {
			result = nil
		}
	}()
	if threads == 0 {
		threads = 1
	}
	if err = o.SetIntraOpNumThreads(threads); err != nil {
		return nil, err
	}
	if err = o.SetInterOpNumThreads(1); err != nil {
		return nil, err
	}
	t.session, err = ort.NewDynamicAdvancedSessionWithONNXData(model, []string{"buckets"}, []string{"vectors", "scores"}, o)
	if err != nil {
		return nil, fmt.Errorf("load trained POS model: %w", err)
	}
	return t, nil
}

func (t *tagger) tag(tokens []Token, features [][6]uint64) (err error) {
	if len(tokens) == 0 {
		return nil
	}
	flat := make([]int64, 0, len(tokens)*24)
	for _, row := range features {
		for j, f := range row {
			for _, hash := range murmur.Hash(f, t.Seeds[j]) {
				flat = append(flat, int64(hash%t.Rows[j]))
			}
		}
	}
	input, err := ort.NewTensor(ort.NewShape(int64(len(tokens)), 6, 4), flat)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, input.Destroy()) }()
	outputs := []ort.Value{nil, nil}
	defer func() {
		for _, v := range outputs {
			if v != nil {
				err = errors.Join(err, v.Destroy())
			}
		}
	}()
	if err = t.session.Run([]ort.Value{input}, outputs); err != nil {
		return err
	}
	scores, ok := outputs[1].(*ort.Tensor[float32])
	if !ok || len(scores.GetData()) != len(tokens)*len(t.Labels) {
		return errors.New("invalid POS output shape/type")
	}
	for i := range tokens {
		row := scores.GetData()[i*50 : (i+1)*50]
		best := 0
		for j, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return errors.New("nonfinite POS score")
			}
			if v > row[best] {
				best = j
			}
		}
		tokens[i].Tag = t.Labels[best]
	}
	return nil
}
