// Package murmur implements the feature hash used by the POS tagger.
package murmur

import "math/bits"

// Hash matches Thinc's MurmurHash3_x86_128_uint64, which despite its name
// implements the x64 variant and splits its two uint64 outputs into four words.
// Adapted from Thinc (MIT); MurmurHash3 is public domain, Austin Appleby.
func Hash(value uint64, seed uint32) [4]uint32 {
	h1 := bits.RotateLeft64(value*0x87c37b91114253d5, 31) * 0x4cf5ad432745937f
	h1 ^= uint64(seed) ^ 8
	h2 := uint64(seed) ^ 8
	h1 += h2
	h2 += h1
	mix := func(h uint64) uint64 {
		h ^= h >> 33
		h *= 0xff51afd7ed558ccd
		h ^= h >> 33
		h *= 0xc4ceb9fe1a85ec53
		return h ^ (h >> 33)
	}
	h1, h2 = mix(h1), mix(h2)
	h1 += h2
	h2 += h1
	return [4]uint32{uint32(h1), uint32(h1 >> 32), uint32(h2), uint32(h2 >> 32)}
}
