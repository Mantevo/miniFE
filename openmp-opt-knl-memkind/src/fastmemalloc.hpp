
#ifndef _H_MINIFE_FAST_MALLOC
#define _H_MINIFE_FAST_MALLOC

#include <cstddef>
#include <cstdio>
#include <cinttypes>
#include <limits>
#include "memkind.h"

template <typename T>
class FastMemAllocator {

public:
	typedef T value_type;
	typedef std::size_t size_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T& reference;
	typedef const T& const_reference;
	typedef std::ptrdiff_t difference_type;

    	template<typename U>
    	struct rebind {
        	typedef FastMemAllocator<U> other;
    	};

	FastMemAllocator() {}
	~FastMemAllocator() {}

    	inline pointer address(reference r) { return &r; }
    	inline const_pointer address(const_reference r) { return &r; }

	pointer allocate(size_type n) {
		pointer the_ptr;

		memkind_posix_memalign(MEMKIND_HBW, (void**) &the_ptr, 64, n * sizeof(value_type));
		return the_ptr;
	}

	void deallocate(pointer ptr, size_type n) {
		memkind_free(MEMKIND_HBW, ptr);
	}

	inline size_type max_size() const { 
        	return std::numeric_limits<size_type>::max() / sizeof(T);
 	}

};

#endif
