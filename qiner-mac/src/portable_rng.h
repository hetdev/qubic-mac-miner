#pragma once

// Portable stand-in for the x86 RDRAND intrinsics used across the Qiner miners
// (nonce generation, gamming nonces and the packet dejavu field).
//
// On x86 targets the real <immintrin.h> intrinsics are used, so behaviour there
// is unchanged. On non-x86 targets (Apple Silicon / ARM64, other ports) the
// same intrinsic signatures are backed by a CSPRNG instead:
//   - Darwin/BSD: arc4random_buf
//   - elsewhere:  std::random_device
// For the miner's use (random nonces) any CSPRNG is a drop-in replacement.

#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#else

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#include <stdlib.h> // arc4random_buf

static inline void qiner_fill_random(void* p, std::size_t n)
{
    arc4random_buf(p, n);
}
#else
#include <random>

static inline void qiner_fill_random(void* p, std::size_t n)
{
    static std::random_device rd;
    unsigned char* out = static_cast<unsigned char*>(p);
    std::size_t i = 0;
    while (i < n)
    {
        const unsigned long long v = rd();
        const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
        for (std::size_t j = 0; j < sizeof(v) && i < n; ++j, ++i)
        {
            out[i] = b[j];
        }
    }
}
#endif

static inline int _rdrand64_step(unsigned long long* val)
{
    qiner_fill_random(val, sizeof(*val));
    return 1;
}

static inline int _rdrand32_step(unsigned int* val)
{
    qiner_fill_random(val, sizeof(*val));
    return 1;
}

#endif // architecture switch
