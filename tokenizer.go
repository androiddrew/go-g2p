package g2p

// Tokenizer rules and lexical features follow spaCy 3.8.7 (MIT), with the
// en_core_web_sm 3.8.0 exported data. See THIRD_PARTY.md and retained notices.
import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/dlclark/regexp2"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

type tokenizer struct {
	Rules                      map[string][]map[string]string `json:"rules"`
	Norms                      map[string]string              `json:"norms"`
	BaseNorms                  map[string]string              `json:"base_norms"`
	Symbols                    map[string]uint64              `json:"symbols"`
	Prefix, Suffix, Infix, URL string
	prefix, suffix, infix, url *regexp2.Regexp
	multi                      []string
	digits                     map[rune]int
}

func readJSON(path string, value any) error {
	return readFSJSON(os.DirFS(filepath.Dir(path)), filepath.Base(path), value)
}

func readFSJSON(fsys fs.FS, name string, value any) error {
	data, err := fs.ReadFile(fsys, name)
	if err != nil {
		return err
	}
	if err = json.Unmarshal(data, value); err != nil {
		return fmt.Errorf("decode %s: %w", name, err)
	}
	return nil
}

func regex(pattern string, flags regexp2.RegexOptions) (*regexp2.Regexp, error) {
	// Python's eight-digit Unicode escape denotes one scalar, not a UTF-16 pair.
	pattern = regexp.MustCompile(`\\U[0-9A-Fa-f]{8}`).ReplaceAllStringFunc(pattern, func(s string) string {
		n, _ := strconv.ParseUint(s[2:], 16, 32)
		return string(rune(n))
	})
	r, err := regexp2.Compile(pattern, flags)
	if err == nil {
		r.MatchTimeout = 2 * time.Second
	}
	return r, err
}

func loadTokenizer(fsys fs.FS) (*tokenizer, error) {
	t := new(tokenizer)
	var u struct {
		Version string
		Digits  map[rune]int
	}
	if err := readFSJSON(fsys, "unicode.json", &u); err != nil {
		return nil, err
	}
	if u.Version != "15.0.0" || len(u.Digits) < 600 {
		return nil, fmt.Errorf("unsupported Unicode metadata: %s", u.Version)
	}
	t.digits = u.Digits
	if err := readFSJSON(fsys, "tokenizer.json", t); err != nil {
		return nil, err
	}
	if len(t.Rules) == 0 || len(t.Symbols) == 0 || t.Prefix == "" || t.Infix == "" {
		return nil, fmt.Errorf("incomplete tokenizer metadata")
	}
	for _, rule := range t.Rules {
		for _, part := range rule {
			if part["65"] == "" {
				return nil, fmt.Errorf("empty tokenizer exception")
			}
		}
	}
	var err error
	for _, item := range []struct {
		pattern string
		dest    **regexp2.Regexp
		flags   regexp2.RegexOptions
	}{
		{t.Prefix, &t.prefix, 0}, {t.Suffix, &t.suffix, 0}, {t.Infix, &t.infix, 0}, {t.URL, &t.url, regexp2.IgnoreCase},
	} {
		*item.dest, err = regex(item.pattern, item.flags)
		if err != nil {
			return nil, err
		}
	}
	for key := range t.Rules {
		if strings.ContainsAny(key, " \t\n") && strings.TrimSpace(key) != "" {
			t.multi = append(t.multi, key)
		}
	}
	sort.Slice(t.multi, func(i, j int) bool {
		if len(t.multi[i]) == len(t.multi[j]) {
			return t.multi[i] < t.multi[j]
		}
		return len(t.multi[i]) > len(t.multi[j])
	})
	return t, nil
}

func (t *tokenizer) exception(text string) []Token {
	result := []Token{}
	for _, part := range t.Rules[text] {
		x := Token{Text: part["65"], head: true}
		if n, ok := part["67"]; ok {
			x.norm = &n
		}
		result = append(result, x)
	}
	return result
}

func (t *tokenizer) split(text string) ([]Token, error) {
	// Peel affixes iteratively, avoiding recursion for adversarial punctuation.
	var left, right []Token
	for text != "" {
		if _, ok := t.Rules[text]; ok {
			left = append(left, t.exception(text)...)
			text = ""
			break
		}
		runes := []rune(text)
		m, err := t.prefix.FindStringMatch(text)
		if err != nil {
			return nil, err
		}
		pre := 0
		if m != nil && m.Index == 0 {
			pre = m.Length
		}
		if pre > 0 {
			if _, ok := t.Rules[string(runes[pre:])]; ok {
				left = append(left, Token{Text: string(runes[:pre]), head: true})
				text = string(runes[pre:])
				continue
			}
		}
		m, err = t.suffix.FindStringMatch(string(runes[pre:]))
		if err != nil {
			return nil, err
		}
		suf := 0
		if m != nil {
			suf = m.Length
		}
		if suf > 0 {
			if _, ok := t.Rules[string(runes[:len(runes)-suf])]; ok {
				right = append(right, Token{Text: string(runes[len(runes)-suf:]), head: true})
				text = string(runes[:len(runes)-suf])
				continue
			}
		}
		if pre > 0 || suf > 0 {
			if pre > 0 {
				left = append(left, Token{Text: string(runes[:pre]), head: true})
			}
			if suf > 0 {
				right = append(right, Token{Text: string(runes[len(runes)-suf:]), head: true})
			}
			text = string(runes[pre : len(runes)-suf])
			continue
		}
		m, err = t.url.FindStringMatch(text)
		if err != nil {
			return nil, err
		}
		if m != nil && m.Index == 0 {
			left = append(left, Token{Text: text, head: true})
			text = ""
			break
		}
		last := 0
		for m, err = t.infix.FindStringMatch(text); m != nil && err == nil; m, err = t.infix.FindNextMatch(m) {
			if m.Index == 0 {
				continue
			}
			if m.Index > last {
				left = append(left, Token{Text: string(runes[last:m.Index]), head: true})
			}
			if m.Length > 0 {
				left = append(left, Token{Text: m.String(), head: true})
			}
			last = m.Index + m.Length
		}
		if err != nil {
			return nil, err
		}
		if last < len(runes) {
			left = append(left, Token{Text: string(runes[last:]), head: true})
		}
		text = ""
	}
	for i := len(right) - 1; i >= 0; i-- {
		left = append(left, right[i])
	}
	return left, nil
}

func pySpace(r rune) bool   { return unicode.IsSpace(r) || r >= 0x1c && r <= 0x1f }
func lower(s string) string { return cases.Lower(language.Und).String(s) }
func upper(s string) string { return cases.Upper(language.Und).String(s) }
func capitalize(s string) string {
	r := []rune(lower(s))
	if len(r) == 0 {
		return s
	}
	return upper(string(r[:1])) + string(r[1:])
}
func all(s string, predicate func(rune) bool) bool {
	for _, r := range s {
		if !predicate(r) {
			return false
		}
	}
	return true
}
func has(set string, r rune) bool { return strings.ContainsRune(set, r) }

func (t *tokenizer) tokenize(text string) ([]Token, error) {
	result := []Token{}
	runes := []rune(text)
	for i := 0; i < len(runes); {
		j := i + 1
		space := pySpace(runes[i])
		for j < len(runes) && pySpace(runes[j]) == space {
			j++
		}
		if space {
			if len(result) > 0 && runes[i] == ' ' {
				result[len(result)-1].Whitespace = " "
				i++
			}
			if i < j {
				part := string(runes[i:j])
				if _, ok := t.Rules[part]; ok {
					result = append(result, t.exception(part)...)
				} else {
					result = append(result, Token{Text: part, head: true})
				}
			}
		} else {
			parts, err := t.split(string(runes[i:j]))
			if err != nil {
				return nil, err
			}
			result = append(result, parts...)
		}
		i = j
	}
	// spaCy's second pass recognizes exceptions containing spaces (e.g. "a. m.").
	for i := 0; i < len(result); i++ {
		for _, key := range t.multi {
			joined := ""
			end := i
			for end < len(result) && len(joined) < len(key) {
				if end > i {
					joined += result[end-1].Whitespace
				}
				joined += result[end].Text
				end++
			}
			if joined != key {
				continue
			}
			replacement := t.exception(key)
			if len(replacement) == 0 {
				continue
			}
			replacement[len(replacement)-1].Whitespace = result[end-1].Whitespace
			next := append([]Token{}, result[:i]...)
			next = append(next, replacement...)
			next = append(next, result[end:]...)
			result = next
			i += len(replacement) - 1
			break
		}
	}
	return result, nil
}

// MurmurHash64A, seed 1, used by spaCy's StringStore (Austin Appleby, public domain).
func (t *tokenizer) stringID(text string) uint64 {
	if text == "" {
		return 0
	}
	if id, ok := t.Symbols[text]; ok {
		return id
	}
	data := []byte(text)
	const m uint64 = 0xc6a4a7935bd1e995
	h := uint64(1) ^ uint64(len(data))*m
	for len(data) >= 8 {
		k := binary.LittleEndian.Uint64(data) * m
		k ^= k >> 47
		k *= m
		h = (h ^ k) * m
		data = data[8:]
	}
	if len(data) > 0 {
		var k uint64
		for i, b := range data {
			k |= uint64(b) << (8 * i)
		}
		h = (h ^ k) * m
	}
	h ^= h >> 47
	h *= m
	return h ^ (h >> 47)
}

func (t *tokenizer) features(tokens []Token) [][6]uint64 {
	result := make([][6]uint64, len(tokens))
	for i, tk := range tokens {
		text := []rune(tk.Text)
		n := ""
		if tk.norm != nil {
			n = *tk.norm
		} else if v, ok := t.Norms[strconv.FormatUint(t.stringID(tk.Text), 10)]; ok {
			n = v
		} else if v, ok := t.BaseNorms[tk.Text]; ok {
			n = v
		} else {
			n = lower(tk.Text)
		}
		shape := []rune{}
		var last rune
		repeat := 0
		for _, r := range text {
			mapped := r
			if unicode.IsLetter(r) {
				mapped = 'x'
				if unicode.IsUpper(r) {
					mapped = 'X'
				}
			} else if _, ok := t.digits[r]; ok {
				mapped = 'd'
			}
			if mapped == last {
				repeat++
			} else {
				repeat = 0
			}
			last = mapped
			if repeat < 4 {
				shape = append(shape, mapped)
			}
		}
		if len(text) >= 100 {
			shape = []rune("LONG")
		}
		start := max(0, len(text)-3)
		result[i] = [6]uint64{t.stringID(n), t.stringID(string(text[:1])), t.stringID(string(text[start:])), t.stringID(string(shape)), 0, 0}
		if tk.Whitespace != "" {
			result[i][4] = 1
		}
		if all(tk.Text, pySpace) {
			result[i][5] = 1
		}
	}
	return result
}
