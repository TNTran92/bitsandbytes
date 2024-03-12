
#include <hip/hip_runtime.h>
#include <BinSearch.h>

#ifndef common
#define common

#ifdef __HIP_PLATFORM_AMD__
// Compiled with HIP-Clang
#endif

using namespace BinSearch;

#define BLOCK_SIZE 16384

struct quantize_block_args {
    BinAlgo<Scalar, float, Direct2> *bin_searcher;
    float *code;
    float *A;
    float *absmax;
    unsigned char *out;
    long long block_end;
    long long block_idx;
    long long threadidx;
		long long blocksize;
};


void quantize_block(const quantize_block_args& args);

#endif
