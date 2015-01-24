#include <sys/fcntl.h>
#include <stdio.h>
#include "spooky-c.h"
#include "map.h"

#define use_value(x) asm volatile("" :: "r" (x) : "memory")

enum {
	ITER = 10,
};

int main(int ac, char **av)
{
	while (*++av) {
		unsigned long start, end;
		uint64_t h1, h2;
		struct spooky_state state;
		size_t size;
		char *map = mapfile(*av, O_RDONLY, &size);
		if (!map) {
			perror(*av);
			continue;
		}
		
		int i;
		for (i = 0; i < size; i += 64)
			use_value(((volatile char *)map)[i]);

		spooky_init(&state, 0x123456789abcdef, 0xfedcba987654321);
		spooky_update(&state, map, size);
		spooky_final(&state, &h1, &h2);

		start = __builtin_ia32_rdtsc();
		for (i = 0; i < ITER; i++) {
			spooky_init(&state, 0x123456789abcdef, 0xfedcba987654321);
			spooky_update(&state, map, size);
			spooky_final(&state, &h1, &h2);
		}
		end = __builtin_ia32_rdtsc();

		printf("%s: %016llx%016llx [%f c/b]\n", *av, 
		        (unsigned long long)h1, 
		        (unsigned long long)h2,
			((end - start) / ITER) / (double)size);

		unmap_file(map, size);
	}
	return 0;
}

