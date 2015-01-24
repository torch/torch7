![spooky-c] (http://halobates.de/spooky-c.png)

This is a C version of Bob Jenkin's spooky hash. The only advantage over
Bob's original version is that it is in C, not C++ and comes with
some test and benchmark code.

This is a very competitive hash function, but is somewhat unportable
(64bit little endian only). It's more portable than some of the 
contenders like CityHash.

Quoting Bobs original description:

 SpookyHash: a 128-bit noncryptographic hash function
 By Bob Jenkins, public domain

	Oct 31 2010: alpha, framework + SpookyHash::Mix appears right
	Oct 31 2011: alpha again, Mix only good to 2^^69 but rest appears right
	Oct 11 2011: C version ported by Andi Kleen (andikleen@github)
	Dec 31 2011: beta, improved Mix, tested it for 2-bit deltas
	Feb  2 2012: production, same bits as beta
	Feb  5 2012: adjusted definitions of uint* to be more portable
	Mar 30 2012: 3 bytes/cycle, not 4.  Alpha was 4 but wasn't thorough enough.
	Apr 27 2012: C version updated by Ziga Zupanec (agiz@github)
 
 Up to 3 bytes/cycle for long messages.  Reasonably fast for short messages.
 All 1 or 2 bit deltas achieve avalanche within 1% bias per output bit.

 This was developed for and tested on 64-bit x86-compatible processors.
 It assumes the processor is little-endian.  There is a macro
 controlling whether unaligned reads are allowed (by default they are).
 This should be an equally good hash on big-endian machines, but it will
 compute different results on them than on little-endian machines.

 Google's CityHash has similar specs to SpookyHash, and CityHash is faster
 on some platforms.  MD4 and MD5 also have similar specs, but they are orders
 of magnitude slower.  CRCs are two or more times slower, but unlike 
 SpookyHash, they have nice math for combining the CRCs of pieces to form 
 the CRCs of wholes.  There are also cryptographic hashes, but those are even 
 slower than MD5.
