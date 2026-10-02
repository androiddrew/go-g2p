package g2p

// Misaki's preprocessing, grouping, stress and right-to-left context resolution.
// Adapted under Apache-2.0; see THIRD_PARTY.md.
import (
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/dlclark/regexp2"
)

var links = regexp.MustCompile(`\[([^\]]+)\]\(([^\)]*)\)`)

type feature struct {
	start, end int
	value      string
}

func preprocess(text string) (string, []feature) {
	text = strings.TrimLeftFunc(text, pySpace)
	var b strings.Builder
	features := []feature{}
	last := 0
	for _, m := range links.FindAllStringSubmatchIndex(text, -1) {
		b.WriteString(text[last:m[0]])
		label, value := text[m[2]:m[3]], text[m[4]:m[5]]
		start := utf8.RuneCountInString(b.String())
		b.WriteString(label)
		features = append(features, feature{start, start + utf8.RuneCountInString(label), value})
		last = m[1]
	}
	b.WriteString(text[last:])
	return b.String(), features
}
func applyFeatures(tokens []Token, features []feature) {
	for _, f := range features {
		v, err := strconv.ParseFloat(f.value, 64)
		numeric := err == nil && (digits(strings.TrimLeft(f.value, "+-")) || f.value == "0.5" || f.value == "+0.5" || f.value == "-0.5")
		override := len(f.value) > 1 && strings.HasPrefix(f.value, "/") && strings.HasSuffix(f.value, "/")
		flags := len(f.value) > 1 && strings.HasPrefix(f.value, "#") && strings.HasSuffix(f.value, "#")
		first := true
		for i := range tokens {
			t := &tokens[i]
			if t.End <= f.start || t.Start >= f.end {
				continue
			}
			if numeric {
				t.stress = ptr(v)
			} else if override {
				t.head = first
				p := ""
				if first {
					p = strings.Trim(f.value, "/")
				}
				t.Phonemes = ptr(p)
				t.Rating = 5
				t.Source = "override"
				first = false
			} else if flags {
				t.numFlags = strings.Trim(f.value, "#")
			}
		}
	}
}
func merge(tokens []Token, phones bool) Token {
	t := tokens[0]
	t.Text = ""
	t.Whitespace = tokens[len(tokens)-1].Whitespace
	t.End = tokens[len(tokens)-1].End
	t.alias = ""
	t.Phonemes = nil
	var b, p strings.Builder
	stresses := map[float64]bool{}
	flags := map[rune]bool{}
	currency := ""
	rating := 5
	source := ""
	weight := -1
	for i, x := range tokens {
		b.WriteString(x.Text)
		if i < len(tokens)-1 {
			b.WriteString(x.Whitespace)
		}
		w := 0
		for _, r := range x.Text {
			w++
			if string(r) != lower(string(r)) {
				w++
			}
		}
		if w > weight {
			weight = w
			t.Tag = x.Tag
		}
		if x.stress != nil {
			stresses[*x.stress] = true
		}
		for _, r := range x.numFlags {
			flags[r] = true
		}
		if x.currency > currency {
			currency = x.currency
		}
		rating = min(rating, x.Rating)
		if x.Source != "" {
			if source == "" {
				source = x.Source
			} else if source != x.Source {
				source = "mixed"
			}
		}
		if phones {
			if x.prespace && p.Len() > 0 && x.Phonemes != nil && *x.Phonemes != "" {
				r, _ := utf8.DecodeLastRuneInString(p.String())
				if !pySpace(r) {
					p.WriteByte(' ')
				}
			}
			if x.Phonemes == nil {
				p.WriteString("❓")
			} else {
				p.WriteString(*x.Phonemes)
			}
		}
	}
	t.Text = b.String()
	if phones {
		t.Phonemes = ptr(p.String())
	}
	t.stress = nil
	if len(stresses) == 1 {
		for s := range stresses {
			t.stress = ptr(s)
		}
	}
	t.currency = currency
	t.Rating = rating
	t.Source = source
	t.numFlags = ""
	ordered := []rune{}
	for r := range flags {
		ordered = append(ordered, r)
	}
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	t.numFlags = string(ordered)
	return t
}

var subtokenPattern = `^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}|\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|[^-_\p{L}'‘’\d]|['‘’]+$`
var punctTags = map[string]bool{".": true, ",": true, "-LRB-": true, "-RRB-": true, "``": true, `""`: true, "''": true, ":": true, "$": true, "#": true, "NFP": true}
var punctPhones = map[string]string{"-LRB-": "(", "-RRB-": ")", "``": "“", `""`: "”", "''": "”"}

func retokenize(tokens []Token) ([][]Token, error) {
	rx, err := regex(subtokenPattern, regexp2.None)
	if err != nil {
		return nil, err
	}
	words := [][]Token{}
	grouped := []bool{}
	currency := ""
	for i, t := range tokens {
		parts := []Token{}
		if t.alias == "" && t.Phonemes == nil {
			for m, e := rx.FindStringMatch(t.Text); m != nil || e != nil; m, e = rx.FindNextMatch(m) {
				if e != nil {
					return nil, e
				}
				p := t
				p.Text = m.String()
				p.Whitespace = ""
				p.head = true
				p.prespace = false
				p.Start = t.Start + m.Index
				p.End = p.Start + m.Length
				parts = append(parts, p)
			}
		} else {
			parts = append(parts, t)
		}
		if len(parts) == 0 {
			continue
		}
		parts[len(parts)-1].Whitespace = t.Whitespace
		for j := range parts {
			p := parts[j]
			if p.alias != "" || p.Phonemes != nil {
			} else if _, ok := currencies[p.Text]; p.Tag == "$" && ok {
				currency = p.Text
				setPronunciation(&p, known("", 4, "currency"))
			} else if p.Tag == ":" && (p.Text == "-" || p.Text == "–") {
				setPronunciation(&p, known("—", 3, "punctuation"))
			} else if punctTags[p.Tag] && !all(lower(p.Text), func(r rune) bool { return r >= 'a' && r <= 'z' }) {
				ps, ok := punctPhones[p.Tag]
				if !ok {
					for _, r := range p.Text {
						if has(puncts, r) {
							ps += string(r)
						}
					}
				}
				setPronunciation(&p, known(ps, 4, "punctuation"))
			} else if currency != "" {
				if p.Tag != "CD" {
					currency = ""
				} else if j+1 == len(parts) && (i+1 == len(tokens) || tokens[i+1].Tag != "CD") {
					p.currency = currency
				}
			} else if j > 0 && j < len(parts)-1 && p.Text == "2" {
				left, _ := utf8.DecodeLastRuneInString(parts[j-1].Text)
				right, _ := utf8.DecodeRuneInString(parts[j+1].Text)
				if unicode.IsLetter(left) && unicode.IsLetter(right) {
					p.alias = "to"
				}
			}
			last := len(words) - 1
			if p.alias != "" || p.Phonemes != nil {
				words = append(words, []Token{p})
				grouped = append(grouped, false)
			} else if last >= 0 && grouped[last] && words[last][len(words[last])-1].Whitespace == "" {
				p.head = false
				words[last] = append(words[last], p)
			} else {
				words = append(words, []Token{p})
				grouped = append(grouped, p.Whitespace == "")
			}
		}
	}
	return words, nil
}
func nextContext(ctx tokenContext, t Token) tokenContext {
	if t.Phonemes != nil {
		for _, r := range *t.Phonemes {
			if has(nonQuotePuncts, r) {
				ctx.vowel = nil
				break
			}
			if has(vowels, r) || has(consonants, r) {
				ctx.vowel = ptr(has(vowels, r))
				break
			}
		}
	}
	ctx.to = t.Text == "to" || t.Text == "To" || t.Text == "TO" && (t.Tag == "TO" || t.Tag == "IN")
	return ctx
}
func resolve(tokens []Token) {
	t := merge(tokens, false)
	classes := map[int]bool{}
	for _, r := range t.Text {
		if has(junks, r) {
			continue
		}
		k := 2
		if unicode.IsLetter(r) {
			k = 0
		} else if r >= '0' && r <= '9' {
			k = 1
		}
		classes[k] = true
	}
	prespace := strings.ContainsAny(t.Text, " /") || len(classes) > 1
	type item struct {
		primary       bool
		weight, index int
	}
	indices := []item{}
	primaries := 0
	for i := range tokens {
		t := &tokens[i]
		if t.Phonemes == nil {
			if i == len(tokens)-1 && utf8.RuneCountInString(t.Text) == 1 && strings.Contains(nonQuotePuncts, t.Text) {
				setPronunciation(t, known(t.Text, 3, "punctuation"))
			} else if all(t.Text, func(r rune) bool { return has(junks, r) }) {
				setPronunciation(t, known("", 3, "separator"))
			}
		} else if i > 0 {
			t.prespace = prespace
		}
		if t.Phonemes != nil && *t.Phonemes != "" {
			p := *t.Phonemes
			weight := 0
			for _, r := range p {
				weight++
				if has("AIOQWYʤʧ", r) {
					weight++
				}
			}
			primary := strings.Contains(p, "ˈ")
			if primary {
				primaries++
			}
			indices = append(indices, item{primary, weight, i})
		}
	}
	if prespace {
		return
	}
	if len(indices) == 2 && utf8.RuneCountInString(tokens[indices[0].index].Text) == 1 {
		i := indices[1].index
		tokens[i].Phonemes = ptr(stress(*tokens[i].Phonemes, ptr(-0.5)))
		return
	}
	if len(indices) < 2 || primaries <= (len(indices)+1)/2 {
		return
	}
	sort.Slice(indices, func(i, j int) bool {
		a, b := indices[i], indices[j]
		if a.primary != b.primary {
			return !a.primary
		}
		if a.weight != b.weight {
			return a.weight < b.weight
		}
		return a.index < b.index
	})
	for _, x := range indices[:len(indices)/2] {
		tokens[x.index].Phonemes = ptr(stress(*tokens[x.index].Phonemes, ptr(-0.5)))
	}
}
