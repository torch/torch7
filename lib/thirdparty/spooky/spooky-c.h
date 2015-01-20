// SpookyHash: a 128-bit noncryptographic hash function
// By Bob Jenkins, public domain
//   Oct 31 2010: alpha, framework + SpookyHash::Mix appears right
//   Oct 11 2011: C version ported by Andi Kleen (andikleen@github)
//   Oct 31 2011: alpha again, Mix only good to 2^^69 but rest appears right
//   Dec 31 2011: beta, improved Mix, tested it for 2-bit deltas
//   Feb  2 2012: production, same bits as beta
//   Feb  5 2012: adjusted definitions of uint* to be more portable
//   Mar 30 2012: 3 bytes/cycle, not 4.  Alpha was 4 but wasn't thorough enough.
//   Apr 27 2012: C version updated by Ziga Zupanec ziga.zupanec@gmail.com (agiz@github)
//
// Up to 3 bytes/cycle for long messages.  Reasonably fast for short messages.
// All 1 or 2 bit deltas achieve avalanche within 1% bias per output bit.
//
// This was developed for and tested on 64-bit x86-compatible processors.
// It assumes the processor is little-endian.  There is a macro
// controlling whether unaligned reads are allowed (by default they are).
// This should be an equally good hash on big-endian machines, but it will
// compute different results on them than on little-endian machines.
//
// Google's CityHash has similar specs to SpookyHash, and CityHash is faster
// on some platforms.  MD4 and MD5 also have similar specs, but they are orders
// of magnitude slower.  CRCs are two or more times slower, but unlike
// SpookyHash, they have nice math for combining the CRCs of pieces to form
// the CRCs of wholes.  There are also cryptographic hashes, but those are even
// slower than MD5.
//

#ifndef _SPOOKY_C_H_
#define _SPOOKY_C_H_

#include <stdint.h>
#include <stddef.h>

#define SC_NUMVARS		12
#define SC_BLOCKSIZE	(8 * SC_NUMVARS)
#define SC_BUFSIZE		(2 * SC_BLOCKSIZE)

struct spooky_state
{
	uint64_t m_data[2 * SC_NUMVARS];
	uint64_t m_state[SC_NUMVARS];
	size_t m_length;
	unsigned char m_remainder;
};

void spooky_shorthash
(
	const void *message,
	size_t length,
	uint64_t *hash1,
	uint64_t *hash2
);

void spooky_init
(
	struct spooky_state *state,
	uint64_t hash1,
	uint64_t hash2
);

void spooky_update
(
	struct spooky_state *state,
	const void *msg,
	size_t len
);

void spooky_final
(
	struct spooky_state *state,
	uint64_t *hash1,
	uint64_t *hash2
);

//hash1/2 doubles as input parameter for seed1/2 and output for hash1/2
void spooky_hash128
(
	const void *message,
	size_t length,
	uint64_t *hash1,
	uint64_t *hash2
);

uint64_t spooky_hash64
(
	const void *message,
	size_t len,
	uint64_t seed
);

uint32_t spooky_hash32
(
	const void *message,
	size_t len,
	uint32_t seed
);


#endif // #define _SPOOKY_C_H_
