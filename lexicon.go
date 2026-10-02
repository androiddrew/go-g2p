package g2p

// Adapted from hexgrad/misaki en.py, Apache-2.0, pinned in THIRD_PARTY.md.
import (
	"encoding/json"
	"fmt"
	"io/fs"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

const vowels = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ"
const consonants = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ"
const taus = "AIOWYiuæɑəɛɪɹʊʌ"
const puncts = ";:,.!?—…\"“”"
const nonQuotePuncts = ";:,.!?—…"
const junks = "',-._‘’/"

type pronunciation struct {
	phones string
	rating int
	source string
	known  bool
}

func known(ps string, rating int, source string) pronunciation {
	return pronunciation{ps, rating, source, true}
}

type tokenContext struct {
	vowel *bool
	to    bool
}
type entry struct {
	single   *string
	variants map[string]*string
}

func (e *entry) UnmarshalJSON(data []byte) error {
	if len(data) > 0 && data[0] == '"' {
		return json.Unmarshal(data, &e.single)
	}
	if err := json.Unmarshal(data, &e.variants); err != nil {
		return err
	}
	if _, ok := e.variants["DEFAULT"]; !ok {
		return fmt.Errorf("dictionary variant lacks DEFAULT")
	}
	return nil
}

type lexicon struct {
	gold, silver map[string]entry
	british      bool
	digits       map[rune]int
}

func loadLexicon(fsys fs.FS, dialect string) (*lexicon, error) {
	l := &lexicon{british: dialect == "gb"}
	for _, item := range []struct {
		name string
		dest *map[string]entry
	}{{"gold", &l.gold}, {"silver", &l.silver}} {
		if err := readFSJSON(fsys, dialect+"_"+item.name+".json", item.dest); err != nil {
			return nil, err
		}
		if len(*item.dest) == 0 {
			return nil, fmt.Errorf("empty %s dictionary", item.name)
		}
		grown := map[string]entry{}
		for k, v := range *item.dest {
			if utf8.RuneCountInString(k) < 2 {
				continue
			}
			if k == lower(k) && k != capitalize(k) {
				grown[capitalize(k)] = v
			} else if k == capitalize(lower(k)) {
				grown[lower(k)] = v
			}
		}
		for k, v := range *item.dest {
			grown[k] = v
		}
		*item.dest = grown
	}
	for _, key := range []string{"am", "to"} {
		if l.gold[key].single == nil {
			return nil, fmt.Errorf("missing required dictionary pronunciation %q", key)
		}
	}
	for _, key := range []string{"DEFAULT", "VBD"} {
		if l.gold["used"].variants[key] == nil {
			return nil, fmt.Errorf("missing required used/%s pronunciation", key)
		}
	}
	return l, nil
}
func stress(ps string, s *float64) string {
	if s == nil {
		return ps
	}
	v := *s
	if v < -1 {
		return strings.NewReplacer("ˈ", "", "ˌ", "").Replace(ps)
	}
	if v == -1 || (v == 0 || v == -0.5) && strings.Contains(ps, "ˈ") {
		return strings.ReplaceAll(strings.ReplaceAll(ps, "ˌ", ""), "ˈ", "ˌ")
	}
	if (v == 0 || v == 0.5 || v == 1) && !strings.ContainsAny(ps, "ˈˌ") {
		return insertStress(ps, 'ˌ')
	}
	if v >= 1 && !strings.Contains(ps, "ˈ") && strings.Contains(ps, "ˌ") {
		return strings.ReplaceAll(ps, "ˌ", "ˈ")
	}
	if v > 1 && !strings.ContainsAny(ps, "ˈˌ") {
		return insertStress(ps, 'ˈ')
	}
	return ps
}
func insertStress(ps string, mark rune) string {
	for i, r := range ps {
		if has(vowels, r) {
			return ps[:i] + string(mark) + ps[i:]
		}
	}
	return ps
}
func ptr[T any](v T) *T { return &v }
func parentTag(tag string) string {
	for _, p := range []struct{ prefix, value string }{{"VB", "VERB"}, {"NN", "NOUN"}, {"ADV", "ADV"}, {"RB", "ADV"}, {"ADJ", "ADJ"}, {"JJ", "ADJ"}} {
		if strings.HasPrefix(tag, p.prefix) {
			return p.value
		}
	}
	return tag
}
func letters(s string) bool { return s != "" && all(s, unicode.IsLetter) }
func asciiLex(r rune) bool {
	return r == '\'' || r == '-' || r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z'
}
func (l *lexicon) isKnown(word string) bool {
	if _, ok := l.gold[word]; ok {
		return true
	}
	if _, ok := l.silver[word]; ok {
		return true
	}
	if _, ok := symbols[word]; ok {
		return true
	}
	if !letters(word) || !all(word, asciiLex) {
		return false
	}
	r := []rune(word)
	if len(r) == 1 {
		return true
	}
	if _, ok := l.gold[lower(word)]; word == upper(word) && ok {
		return true
	}
	return string(r[1:]) == upper(string(r[1:]))
}
func (l *lexicon) acronym(word string) pronunciation {
	ps := ""
	for _, c := range word {
		if !unicode.IsLetter(c) {
			continue
		}
		e, ok := l.gold[upper(string(c))]
		if !ok || e.single == nil {
			return pronunciation{}
		}
		ps += *e.single
	}
	ps = stress(ps, ptr(0.0))
	if i := strings.LastIndex(ps, "ˌ"); i >= 0 {
		ps = ps[:i] + "ˈ" + ps[i+len("ˌ"):]
	}
	return known(ps, 3, "acronym")
}
func (l *lexicon) lookup(word, tag string, s *float64, ctx tokenContext) pronunciation {
	nnp := false
	if word == upper(word) {
		if _, ok := l.gold[word]; !ok {
			word = lower(word)
			nnp = tag == "NNP"
		}
	}
	e, ok := l.gold[word]
	rating, source := 4, "gold"
	if !ok && !nnp {
		// A word in neither dictionary leaves e empty, which resolves to no
		// pronunciation below.
		e = l.silver[word]
		rating, source = 3, "silver"
	}
	ps := e.single
	if e.variants != nil {
		if _, yes := e.variants["None"]; yes && ctx.vowel == nil {
			tag = "None"
		} else if _, yes := e.variants[tag]; !yes {
			tag = parentTag(tag)
		}
		var yes bool
		ps, yes = e.variants[tag]
		if !yes {
			ps = e.variants["DEFAULT"]
		}
	}
	if ps == nil || nnp && !strings.Contains(*ps, "ˈ") {
		return l.acronym(word)
	}
	return known(stress(*ps, s), rating, source)
}

var symbols = map[string]string{"%": "percent", "&": "and", "+": "plus", "@": "at"}
var currencies = map[string][2]string{"$": {"dollar", "cent"}, "£": {"pound", "pence"}, "€": {"euro", "cent"}}

func (l *lexicon) special(word, tag string, s *float64, ctx tokenContext) pronunciation {
	if tag == "ADD" && (word == "." || word == "/") {
		alias := "dot"
		if word == "/" {
			alias = "slash"
		}
		return l.lookup(alias, "", ptr(-0.5), ctx)
	}
	if alias, ok := symbols[word]; ok {
		return l.lookup(alias, "", nil, ctx)
	}
	if strings.Contains(strings.Trim(word, "."), ".") && letters(strings.ReplaceAll(word, ".", "")) {
		maxlen := 0
		for _, p := range strings.Split(word, ".") {
			maxlen = max(maxlen, utf8.RuneCountInString(p))
		}
		if maxlen < 3 {
			return l.acronym(word)
		}
	}
	k := lower(word)
	r := func(ps string) pronunciation { return known(ps, 4, "context") }
	if word == "a" || word == "A" {
		if tag == "DT" {
			return r("ɐ")
		}
		return r("ˈA")
	}
	switch k {
	case "am":
		if word != "am" && word != "Am" && word != "AM" {
			break
		}
		if strings.HasPrefix(tag, "NN") {
			return l.acronym(word)
		}
		if ctx.vowel == nil || word != "am" || s != nil && *s > 0 {
			return l.lookup("am", "", nil, tokenContext{})
		}
		return r("ɐm")
	case "an":
		if word == "an" || word == "An" || word == "AN" {
			if word == "AN" && strings.HasPrefix(tag, "NN") {
				return l.acronym(word)
			}
			return r("ɐn")
		}
	case "by":
		if (word == "by" || word == "By" || word == "BY") && parentTag(tag) == "ADV" {
			return r("bˈI")
		}
	case "to":
		if word == "to" || word == "To" || word == "TO" && (tag == "TO" || tag == "IN") {
			if ctx.vowel == nil {
				return l.lookup("to", "", nil, tokenContext{})
			}
			if *ctx.vowel {
				return r("tʊ")
			}
			return r("tə")
		}
	case "in":
		if word == "in" || word == "In" || word == "IN" && tag != "NNP" {
			if ctx.vowel == nil || tag != "IN" {
				return r("ˈɪn")
			}
			return r("ɪn")
		}
	case "the":
		if word == "the" || word == "The" || word == "THE" && tag == "DT" {
			if ctx.vowel != nil && *ctx.vowel {
				return r("ði")
			}
			return r("ðə")
		}
	case "vs", "vs.":
		if tag == "IN" {
			return l.lookup("versus", "", nil, ctx)
		}
	case "used":
		if word == "used" || word == "Used" || word == "USED" {
			v := "DEFAULT"
			if (tag == "VBD" || tag == "JJ") && ctx.to {
				v = "VBD"
			}
			return r(*l.gold["used"].variants[v])
		}
	}
	if word == "I" && tag == "PRP" {
		return r("ˌI")
	}
	return pronunciation{}
}
func (l *lexicon) inflect(p pronunciation, suffix string) pronunciation {
	if !p.known || p.phones == "" {
		return pronunciation{}
	}
	r := []rune(p.phones)
	last := r[len(r)-1]
	extra := ""
	switch suffix {
	case "s":
		extra = "z"
		if has("ptkfθ", last) {
			extra = "s"
		} else if has("szʃʒʧʤ", last) {
			extra = "ᵻz"
			if l.british {
				extra = "ɪz"
			}
		}
	case "ed":
		extra = "d"
		if has("pkfθʃsʧ", last) {
			extra = "t"
		} else if last == 'd' {
			extra = "ᵻd"
			if l.british {
				extra = "ɪd"
			}
		} else if last == 't' {
			extra = "ᵻd"
			if l.british || len(r) < 2 {
				extra = "ɪd"
			} else if has(taus, r[len(r)-2]) {
				r[len(r)-1] = 'ɾ'
			}
		}
	case "ing":
		extra = "ɪŋ"
		if l.british && has("əː", last) {
			return pronunciation{}
		}
		if !l.british && len(r) > 1 && last == 't' && has(taus, r[len(r)-2]) {
			r[len(r)-1] = 'ɾ'
		}
	}
	p.phones = string(r) + extra
	p.source = "morphology"
	return p
}
func (l *lexicon) stem(word, tag string, s *float64, ctx tokenContext, suffix string) pronunciation {
	r := []rune(word)
	n := len(r)
	candidates := []string{}
	switch suffix {
	case "s":
		if n < 3 || !strings.HasSuffix(word, "s") {
			return pronunciation{}
		}
		if !strings.HasSuffix(word, "ss") {
			candidates = append(candidates, string(r[:n-1]))
		}
		if strings.HasSuffix(word, "'s") || n > 4 && strings.HasSuffix(word, "es") && !strings.HasSuffix(word, "ies") {
			candidates = append(candidates, string(r[:n-2]))
		}
		if n > 4 && strings.HasSuffix(word, "ies") {
			candidates = append(candidates, string(r[:n-3])+"y")
		}
	case "ed":
		if n < 4 || !strings.HasSuffix(word, "d") {
			return pronunciation{}
		}
		if !strings.HasSuffix(word, "dd") {
			candidates = append(candidates, string(r[:n-1]))
		}
		if n > 4 && strings.HasSuffix(word, "ed") && !strings.HasSuffix(word, "eed") {
			candidates = append(candidates, string(r[:n-2]))
		}
	case "ing":
		if n < 5 || !strings.HasSuffix(word, "ing") {
			return pronunciation{}
		}
		if n > 5 {
			candidates = append(candidates, string(r[:n-3]))
		}
		candidates = append(candidates, string(r[:n-3])+"e")
		if n > 5 && (r[n-4] == r[n-5] && has("bcdgklmnprstvxz", r[n-4]) || strings.HasSuffix(word, "cking")) {
			candidates = append(candidates, string(r[:n-4]))
		}
	}
	for _, candidate := range candidates {
		if l.isKnown(candidate) {
			return l.inflect(l.lookup(candidate, tag, s, ctx), suffix)
		}
	}
	return pronunciation{}
}
func (l *lexicon) word(word, tag string, s *float64, ctx tokenContext) pronunciation {
	if p := l.special(word, tag, s, ctx); p.known {
		return p
	}
	wl := lower(word)
	r := []rune(word)
	_, gold := l.gold[word]
	_, silver := l.silver[word]
	if len(r) > 1 && letters(strings.ReplaceAll(word, "'", "")) && word != wl && (tag != "NNP" || len(r) > 7) && !gold && !silver && (word == upper(word) || string(r[1:]) == lower(string(r[1:]))) {
		_, g := l.gold[wl]
		_, sv := l.silver[wl]
		ok := g || sv
		for _, suffix := range []string{"s", "ed", "ing"} {
			ok = ok || l.stem(wl, tag, s, ctx, suffix).known
		}
		if ok {
			word = wl
		}
	}
	if l.isKnown(word) {
		return l.lookup(word, tag, s, ctx)
	}
	if strings.HasSuffix(word, "s'") && l.isKnown(strings.TrimSuffix(word, "s'")+"'s") {
		return l.lookup(strings.TrimSuffix(word, "s'")+"'s", tag, s, ctx)
	}
	if strings.HasSuffix(word, "'") && l.isKnown(strings.TrimSuffix(word, "'")) {
		return l.lookup(strings.TrimSuffix(word, "'"), tag, s, ctx)
	}
	for _, suffix := range []string{"s", "ed", "ing"} {
		st := s
		if suffix == "ing" && st == nil {
			st = ptr(0.5)
		}
		if p := l.stem(word, tag, st, ctx, suffix); p.known {
			return p
		}
	}
	return pronunciation{}
}
func (l *lexicon) pronounce(t Token, ctx tokenContext) pronunciation {
	word := t.Text
	if t.alias != "" {
		word = t.alias
	}
	word = norm.NFKC.String(strings.NewReplacer("‘", "'", "’", "'").Replace(word))
	// Decimal digits outside ASCII retain their numeric value after NFKC.
	var b strings.Builder
	for _, r := range word {
		if digit, ok := l.digits[r]; ok {
			r = '0' + rune(digit)
		}
		b.WriteRune(r)
	}
	word = b.String()
	var s *float64
	if word != lower(word) {
		s = ptr(0.5)
		if word == upper(word) {
			s = ptr(2.0)
		}
	}
	p := l.word(word, t.Tag, s, ctx)
	if p.known {
		if unit, ok := currencies[t.currency]; ok {
			extra := l.stem(unit[0]+"s", "", nil, tokenContext{}, "s")
			p.phones += " " + extra.phones
		}
	} else if isNumber(word, t.head) {
		p = l.number(word, t.currency, t.head, t.numFlags)
	}
	if p.known {
		p.phones = stress(p.phones, t.stress)
	}
	return p
}
