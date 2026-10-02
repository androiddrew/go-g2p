package g2p

// English number spelling is independently implemented from arithmetic rules;
// This implementation uses arithmetic rules and has no external number-spelling dependency.
import (
	"strconv"
	"strings"
)

var small = []string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"}
var tens = []string{"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}

func cardinal(n uint64) []string {
	if n < 20 {
		return []string{small[n]}
	}
	if n < 100 {
		result := []string{tens[n/10]}
		if n%10 != 0 {
			result = append(result, small[n%10])
		}
		return result
	}
	if n < 1000 {
		result := []string{small[n/100], "hundred"}
		if n%100 != 0 {
			result = append(result, "and")
			result = append(result, cardinal(n%100)...)
		}
		return result
	}
	for _, scale := range []struct {
		value uint64
		name  string
	}{{1000000000000000000, "quintillion"}, {1000000000000000, "quadrillion"}, {1000000000000, "trillion"}, {1000000000, "billion"}, {1000000, "million"}, {1000, "thousand"}} {
		if n < scale.value {
			continue
		}
		result := append(cardinal(n/scale.value), scale.name)
		rem := n % scale.value
		if rem > 0 {
			if rem < 100 {
				result = append(result, "and")
			}
			result = append(result, cardinal(rem)...)
		}
		return result
	}
	return nil
}
func ordinal(n uint64) []string {
	w := cardinal(n)
	last := w[len(w)-1]
	irregular := map[string]string{"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
	if v, ok := irregular[last]; ok {
		last = v
	} else if strings.HasSuffix(last, "y") {
		last = strings.TrimSuffix(last, "y") + "ieth"
	} else {
		last += "th"
	}
	w[len(w)-1] = last
	return w
}
func year(n uint64) []string {
	if n < 100 || n%1000 < 10 && n >= 1000 {
		return cardinal(n)
	}
	a, b := n/100, n%100
	result := cardinal(a)
	if b == 0 {
		return append(result, "hundred")
	}
	if b < 10 {
		return append(result, "oh", small[b])
	}
	return append(result, cardinal(b)...)
}
func digits(s string) bool {
	return s != "" && all(s, func(r rune) bool { return r >= '0' && r <= '9' })
}
func suffixNumber(word string) (string, string) {
	for _, s := range []string{"ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"} {
		if strings.HasSuffix(word, s) {
			return strings.TrimSuffix(word, s), s
		}
	}
	return word, ""
}
func isNumber(word string, head bool) bool {
	word, _ = suffixNumber(word)
	found := false
	for i, r := range word {
		if r >= '0' && r <= '9' {
			found = true
		} else if r != ',' && r != '.' && !(head && i == 0 && r == '-') {
			return false
		}
	}
	return found
}
func (l *lexicon) number(word, currency string, head bool, flags string) pronunciation {
	word, suffix := suffixNumber(word)
	result := []string{}
	valid := true
	appendWord := func(w string, s *float64) {
		p := l.lookup(w, "", s, tokenContext{})
		if !p.known {
			valid = false
		}
		result = append(result, p.phones)
	}
	extend := func(words []string, first bool) {
		for i, w := range words {
			if w != "and" || strings.Contains(flags, "&") {
				if first && i == 0 && len(words) > 1 && w == "one" && strings.Contains(flags, "a") {
					result = append(result, "ə")
				} else {
					var s *float64
					if w == "point" {
						s = ptr(-2.0)
					}
					appendWord(w, s)
				}
			} else if strings.Contains(flags, "n") && len(result) > 0 {
				result[len(result)-1] += "ən"
			}
		}
	}
	parse := func(s string) uint64 {
		n, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			valid = false
		}
		return n
	}
	negative := strings.HasPrefix(word, "-")
	if negative {
		appendWord("minus", nil)
		word = word[1:]
	}
	_, money := currencies[currency]
	isOrdinal := suffix == "st" || suffix == "nd" || suffix == "rd" || suffix == "th"
	if digits(word) && isOrdinal {
		extend(ordinal(parse(word)), true)
	} else if !negative && len(word) == 4 && !money && digits(word) {
		extend(year(parse(word)), true)
	} else if !head && !strings.Contains(word, ".") {
		num := strings.ReplaceAll(word, ",", "")
		if num == "" {
			return pronunciation{}
		}
		if num[0] == '0' || len(num) > 3 {
			for _, c := range num {
				extend(cardinal(uint64(c-'0')), false)
			}
		} else if len(num) == 3 && !strings.HasSuffix(num, "00") {
			extend(cardinal(uint64(num[0]-'0')), true)
			if num[1] == '0' {
				appendWord("O", ptr(-2.0))
				extend(cardinal(uint64(num[2]-'0')), false)
			} else {
				extend(cardinal(parse(num[1:])), false)
			}
		} else {
			extend(cardinal(parse(num)), true)
		}
	} else if strings.Count(word, ".") > 1 || !head {
		for i, num := range strings.Split(strings.ReplaceAll(word, ",", ""), ".") {
			if num == "" {
				continue
			}
			if num[0] == '0' || len(num) != 2 && strings.Trim(num[1:], "0") != "" {
				for _, c := range num {
					extend(cardinal(uint64(c-'0')), false)
				}
			} else {
				extend(cardinal(parse(num)), i == 0)
			}
		}
	} else if money && (strings.Count(word, ".") == 0 || len(strings.Split(word, ".")[1]) < 3) {
		parts := strings.Split(strings.ReplaceAll(word, ",", ""), ".")
		nums := []uint64{}
		for _, part := range parts {
			n := uint64(0)
			if part != "" {
				n = parse(part)
			}
			nums = append(nums, n)
		}
		start, end := 0, len(nums)
		if end > 1 {
			if nums[1] == 0 {
				end = 1
			} else if nums[0] == 0 {
				start = 1
			}
		}
		for i := start; i < end; i++ {
			if i > start {
				appendWord("and", nil)
			}
			extend(cardinal(nums[i]), i == start)
			unit := currencies[currency][i]
			p := l.lookup(unit, "", nil, tokenContext{})
			if nums[i] != 1 && unit != "pence" {
				p = l.stem(unit+"s", "", nil, tokenContext{}, "s")
			}
			valid = valid && p.known
			result = append(result, p.phones)
		}
	} else {
		word = strings.ReplaceAll(word, ",", "")
		if !strings.Contains(word, ".") {
			if isOrdinal {
				extend(ordinal(parse(word)), true)
			} else {
				extend(cardinal(parse(word)), true)
			}
		} else {
			parts := strings.SplitN(word, ".", 2)
			if parts[0] != "" {
				extend(cardinal(parse(parts[0])), true)
			}
			fraction := parts[1]
			if parts[0] != "" {
				fraction = strings.TrimRight(fraction, "0")
			}
			if fraction != "" || parts[0] == "" {
				appendWord("point", ptr(-2.0))
				for _, c := range fraction {
					extend(cardinal(uint64(c-'0')), false)
				}
			}
		}
	}
	if !valid || len(result) == 0 {
		return pronunciation{}
	}
	p := known(strings.Join(result, " "), 4, "number")
	switch suffix {
	case "s", "'s":
		p = l.inflect(p, "s")
	case "ed", "'d":
		p = l.inflect(p, "ed")
	case "ing":
		p = l.inflect(p, "ing")
	}
	return p
}
