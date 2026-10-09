// metal_bench — host for the Metal bpp9000 walk PoC.
//
// Uses the REFERENCE C++ scorer (qiner-mac/src/score_bpp9000.h) as ground
// truth: it derives the exact per-nonce inputs (root material + mutation
// seeds via K12/random2), runs the CPU walk, then runs the same walk on the
// GPU (kernels.metal) and compares bit-exactly.
//
//   ./metal_bench validate          # 3 cases: golden + 2 random, CPU vs GPU
//   ./metal_bench bench [N]         # N-nonce GPU batch vs CPU subset
//
// Build: ./build.sh   (clang++ ObjC++ + runtime-compiled MSL)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "score_bpp9000.h"

using ProdMiner = score_bpp9000::Miner<
    score_bpp9000::NUMBER_OF_INPUT_NEURONS,
    score_bpp9000::NUMBER_OF_OUTPUT_NEURONS,
    score_bpp9000::SEQUENCE_LENGTH,
    score_bpp9000::WINDOW_WIDTH,
    score_bpp9000::MAX_NUMBER_OF_TICKS,
    score_bpp9000::NUMBER_OF_NEIGHBORS,
    score_bpp9000::POPULATION_THRESHOLD,
    score_bpp9000::NUMBER_OF_MUTATIONS,
    score_bpp9000::SOLUTION_THRESHOLD,
    score_bpp9000::SHIFT_CAP>;

// Must mirror metal::WalkState in kernels.metal (same member order).
struct WalkStateCPP
{
    unsigned char  curLut[1728];
    unsigned char  curInit[64];
    uint16_t       curNeigh[192];
    unsigned char  prevLut[1728];
    unsigned char  prevInit[64];
    uint16_t       prevNeigh[192];
    unsigned char  bestLut[1728];
    unsigned char  bestInit[64];
    uint16_t       bestNeigh[192];
};
static_assert(sizeof(WalkStateCPP) == 6528, "WalkState layout must match kernels.metal");

// Must mirror metal::TGState in kernels.metal (v2) — used for the
// threadgroup-memory length.
struct TGStateCPP
{
    unsigned char  curLut[1728];
    unsigned char  curInit[64];
    uint16_t       curNeigh[192];
    unsigned char  prevLut[1728];
    unsigned char  prevInit[64];
    uint16_t       prevNeigh[192];
    unsigned char  bestLut[1728];
    unsigned char  bestInit[64];
    uint16_t       bestNeigh[192];
    unsigned char  neuronA[64];
    unsigned char  neuronB[64];
    uint32_t       counter;
    uint32_t       failures;
    uint64_t       ticks;
    uint64_t       shift;
    uint32_t       pendingShift;
};
static_assert(sizeof(TGStateCPP) == 6688, "TGState layout must match kernels.metal");

// Must mirror metal::TGState4 (v3) — TGState + shared walk scalars.
struct TGState4CPP
{
    unsigned char  curLut[1728];
    unsigned char  curInit[64];
    uint16_t       curNeigh[192];
    unsigned char  prevLut[1728];
    unsigned char  prevInit[64];
    uint16_t       prevNeigh[192];
    unsigned char  bestLut[1728];
    unsigned char  bestInit[64];
    uint16_t       bestNeigh[192];
    unsigned char  neuronA[64];
    unsigned char  neuronB[64];
    uint32_t       counter;
    uint32_t       failures;
    uint64_t       ticks;
    uint64_t       shift;
    uint32_t       pendingShift;
    uint32_t       curErr;
    uint32_t       curShiftV;
    uint32_t       bestErr;
    uint32_t       bestShiftV;
    uint32_t       accepted;
    uint32_t       takeBest;
};
static_assert(sizeof(TGState4CPP) == 6712, "TGState4 layout must match kernels.metal");

static constexpr uint32_t WALKS_PER_GROUP = 4;

struct WalkParamsCPP
{
    uint32_t controlIndex;
    uint32_t outputIndex;
};

static constexpr size_t SEEDS_PER_NONCE = 1000 * 10;   // MUTATIONS * MAX_L

static bool parseHex32(const char* hex, unsigned char* out)
{
    for (int i = 0; i < 32; ++i)
    {
        unsigned int v;
        if (sscanf(hex + 2 * i, "%2x", &v) != 1) return false;
        out[i] = (unsigned char)v;
    }
    return true;
}

static void canonicalNonce(unsigned char* nonce)
{
    arc4random_buf(nonce, 32);
    const unsigned char L = (unsigned char)((nonce[1] % score_bpp9000::MAX_CHANGES_PER_STEP) + 1);
    const unsigned char mode = (unsigned char)(((nonce[1] >> 4) % 3) + 1);
    nonce[0] = (unsigned char)AlgoType::Bpp9000;
    nonce[1] = (unsigned char)(L | (mode << 4));
    nonce[2] = 0;
}

// Applied root material, exactly like Miner::applyRootMaterial().
static void appliedRoot(ProdMiner& m, std::vector<unsigned char>& lut,
                        std::vector<unsigned char>& start, std::vector<uint16_t>& neigh)
{
    lut.resize(1728);
    start.resize(64);
    neigh.resize(192);
    for (int i = 0; i < 1728; ++i) lut[i] = (unsigned char)(m.rootMaterial.lut[i] % 3);
    for (int n = 0; n < 64; ++n)   start[n] = (unsigned char)(m.rootMaterial.start[n] % 3);
    for (int i = 0; i < 192; ++i)  neigh[i] = (uint16_t)(m.rootMaterial.wire[i] % 64);
}

struct MetalCtx
{
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipe;
};

static bool initMetal(MetalCtx& ctx, const std::string& kernelPath)
{
    ctx.device = MTLCreateSystemDefaultDevice();
    if (!ctx.device) { fprintf(stderr, "No Metal device\n"); return false; }
    ctx.queue = [ctx.device newCommandQueue];

    NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:kernelPath.c_str()]
                                              encoding:NSUTF8StringEncoding error:nil];
    if (!src) { fprintf(stderr, "Cannot read %s\n", kernelPath.c_str()); return false; }

    NSError* err = nil;
    id<MTLLibrary> lib = [ctx.device newLibraryWithSource:src options:nil error:&err];
    if (!lib) { fprintf(stderr, "MSL compile error: %s\n", err.localizedDescription.UTF8String); return false; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"bppWalkSimd4"];
    ctx.pipe = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
    if (!ctx.pipe) { fprintf(stderr, "Pipeline error: %s\n", err.localizedDescription.UTF8String); return false; }
    return true;
}

struct GpuInput
{
    std::vector<unsigned char> lut, start;
    std::vector<uint16_t> neigh;
    std::vector<unsigned long long> seeds;   // SEEDS_PER_NONCE
    uint32_t L, mode;
};

// Runs N walks on the GPU; returns (error, shift) per nonce.
static std::vector<std::pair<uint32_t, uint32_t>>
gpuWalks(MetalCtx& ctx, const unsigned char* target,
         const std::vector<GpuInput>& inputs, uint32_t controlIndex, uint32_t outputIndex)
{
    const NSUInteger n = inputs.size();
    const NSUInteger nPadded = ((n + 3) / 4) * 4;   // 4 walks per threadgroup

    id<MTLBuffer> bTarget = [ctx.device newBufferWithBytes:target length:8928 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bLut    = [ctx.device newBufferWithBytes:inputs[0].lut.data()   length:1728 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bStart  = [ctx.device newBufferWithBytes:inputs[0].start.data() length:64   options:MTLResourceStorageModeShared];
    id<MTLBuffer> bNeigh  = [ctx.device newBufferWithBytes:inputs[0].neigh.data() length:384  options:MTLResourceStorageModeShared];

    std::vector<unsigned long long> seeds(nPadded * SEEDS_PER_NONCE, 0);
    std::vector<uint32_t> params(nPadded * 2, 0);
    for (NSUInteger i = 0; i < nPadded; ++i)
    {
        params[i * 2 + 0] = 1;   // padded slots: harmless walk (L=1, mode=1)
        params[i * 2 + 1] = 1;
    }
    for (NSUInteger i = 0; i < n; ++i)
    {
        memcpy(&seeds[i * SEEDS_PER_NONCE], inputs[i].seeds.data(), SEEDS_PER_NONCE * sizeof(unsigned long long));
        params[i * 2 + 0] = inputs[i].L;
        params[i * 2 + 1] = inputs[i].mode;
    }
    id<MTLBuffer> bSeeds  = [ctx.device newBufferWithBytes:seeds.data()  length:seeds.size() * 8 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bParams = [ctx.device newBufferWithBytes:params.data() length:params.size() * 4 options:MTLResourceStorageModeShared];

    WalkParamsCPP wp{ controlIndex, outputIndex };
    id<MTLBuffer> bShared = [ctx.device newBufferWithBytes:&wp length:sizeof(wp) options:MTLResourceStorageModeShared];

    id<MTLBuffer> bState = [ctx.device newBufferWithLength:n * sizeof(WalkStateCPP) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bOut   = [ctx.device newBufferWithLength:nPadded * 8 options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.pipe];
    [enc setBuffer:bTarget offset:0 atIndex:0];
    [enc setBuffer:bLut    offset:0 atIndex:1];
    [enc setBuffer:bStart  offset:0 atIndex:2];
    [enc setBuffer:bNeigh  offset:0 atIndex:3];
    [enc setBuffer:bSeeds  offset:0 atIndex:4];
    [enc setBuffer:bParams offset:0 atIndex:5];
    [enc setBuffer:bShared offset:0 atIndex:6];
    [enc setBuffer:bState  offset:0 atIndex:7];
    [enc setBuffer:bOut    offset:0 atIndex:8];
    [enc setThreadgroupMemoryLength:WALKS_PER_GROUP * sizeof(TGState4CPP) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nPadded / WALKS_PER_GROUP, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32 * WALKS_PER_GROUP, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    std::vector<std::pair<uint32_t, uint32_t>> out(n);
    const uint32_t* p = (const uint32_t*)bOut.contents;
    for (NSUInteger i = 0; i < n; ++i) out[i] = { p[i * 2], p[i * 2 + 1] };
    return out;
}

int main(int argc, char** argv)
{
    const std::string mode = (argc > 1) ? argv[1] : "validate";
    std::string kernelPath = "kernels.metal";
    std::string taskPath = "../qiner-mac/test/data/bpp9000.task";
    NSUInteger benchN = 256;
    if (mode == "bench")
    {
        if (argc > 2) benchN = (NSUInteger)atoi(argv[2]);
        if (argc > 3) kernelPath = argv[3];
        if (argc > 4) taskPath = argv[4];
    }
    else
    {
        if (argc > 2) kernelPath = argv[2];
        if (argc > 3) taskPath = argv[3];
    }

    // Golden inputs (test/data/gt_production.csv row 0 + the golden mining seed)
    unsigned char miningSeed[32], goldenPub[32], goldenNonce[32];
    parseHex32("8b12add89bc264e01038b61b784e0778a03cd3025e1faeedda6598f197ceccc0", miningSeed);
    parseHex32("039cc1f1560aa96daa994a2b296f22d7f2fc9503ce95321d0b8193079e5f93dc", goldenPub);
    parseHex32("bee1e55fc2c967601827dd80f09a6b0a47ceab37b920b57efbd23c8fdc49e38f", goldenNonce);

    // Reference scorer + shared random2 pool
    std::vector<unsigned char> pool(POOL_VEC_PADDING_SIZE);
    generateRandom2Pool(miningSeed, pool.data());
    ProdMiner m;
    m.setPool(pool.data());
    m.deriveControlOutput(miningSeed);
    if (!m.loadTaskData(taskPath.c_str()))
    {
        fprintf(stderr, "Failed to load task %s\n", taskPath.c_str());
        return 1;
    }
    printf("task loaded | control=%u output=%u\n", m.controlIndex, m.outputIndex);

    unsigned char target[8928];
    for (unsigned long long i = 0; i < score_bpp9000::SEQUENCE_LENGTH; ++i)
        target[i] = m.outputs[i][0];

    MetalCtx ctx;
    if (!initMetal(ctx, kernelPath)) return 1;
    printf("Metal device: %s\n\n", ctx.device.name.UTF8String);

    // ---------------- validate ----------------
    if (mode == "validate")
    {
        struct Case { const char* name; unsigned char pub[32]; unsigned char nonce[32]; };
        std::vector<Case> cases(3);
        cases[0].name = "golden row 0 (expect error 4811)";
        memcpy(cases[0].pub, goldenPub, 32);
        memcpy(cases[0].nonce, goldenNonce, 32);
        cases[1].name = "random 1";
        cases[2].name = "random 2";
        arc4random_buf(cases[1].pub, 32);
        arc4random_buf(cases[2].pub, 32);
        canonicalNonce(cases[1].nonce);
        canonicalNonce(cases[2].nonce);

        bool allOk = true;
        for (auto& c : cases)
        {
            // CPU reference walk
            auto t0 = std::chrono::steady_clock::now();
            const score_bpp9000::Rating cpu = m.computeScore(c.pub, c.nonce);
            double cpuSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            // stage GPU inputs from the same derivation the CPU used
            GpuInput in;
            appliedRoot(m, in.lut, in.start, in.neigh);
            in.seeds.assign(m.mutationSeed, m.mutationSeed + SEEDS_PER_NONCE);
            in.L = c.nonce[1] & 0x0F;
            in.mode = (c.nonce[1] >> 4) & 0x03;

            t0 = std::chrono::steady_clock::now();
            auto gpu = gpuWalks(ctx, target, { in }, m.controlIndex, m.outputIndex)[0];
            double gpuSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            const bool gpuMatch = (gpu.first == cpu.error) && (gpu.second == cpu.shift);
            const bool goldenOk = (strcmp(c.name, "golden row 0 (expect error 4811)") != 0) || (cpu.error == 4811);
            const bool ok = gpuMatch && goldenOk;
            allOk &= ok;
            printf("[%s] %s\n", ok ? "MATCH" : "MISMATCH", c.name);
            printf("    CPU: error=%u shift=%u (%.1fs)  |  GPU: error=%u shift=%u (%.1fs)\n",
                   cpu.error, cpu.shift, cpuSec, gpu.first, gpu.second, gpuSec);
        }
        printf("\n%s\n", allOk ? "ALL CASES BIT-EXACT" : "VALIDATION FAILED");
        return allOk ? 0 : 1;
    }

    // ---------------- bench ----------------
    const NSUInteger n = benchN;
    printf("staging %lu nonces (CPU derivation, no walk)...\n", (unsigned long)n);

    unsigned char pub[32];
    memcpy(pub, goldenPub, 32);
    m.deriveRootMaterial(pub);

    std::vector<GpuInput> inputs(n);
    for (NSUInteger i = 0; i < n; ++i)
    {
        appliedRoot(m, inputs[i].lut, inputs[i].start, inputs[i].neigh);
        unsigned char nonce[32];
        canonicalNonce(nonce);
        m.deriveMutationSeeds(pub, nonce, nullptr);
        inputs[i].seeds.assign(m.mutationSeed, m.mutationSeed + SEEDS_PER_NONCE);
        inputs[i].L = nonce[1] & 0x0F;
        inputs[i].mode = (nonce[1] >> 4) & 0x03;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto gpu = gpuWalks(ctx, target, inputs, m.controlIndex, m.outputIndex);
    double gpuSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // CPU subset for the ratio (a full CPU run of N nonces takes too long)
    const NSUInteger k = (n < 8) ? n : 8;
    t0 = std::chrono::steady_clock::now();
    for (NSUInteger i = 0; i < k; ++i)
    {
        unsigned char nonce[32];
        canonicalNonce(nonce);
        m.computeScore(pub, nonce);
    }
    double cpuSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    printf("\n=== Metal bench: %lu nonces ===\n", (unsigned long)n);
    printf("GPU: %.2fs total -> %.1f nonces/s\n", gpuSec, n / gpuSec);
    printf("CPU: %.2fs for %lu nonces -> %.2f nonces/s (single core)\n", cpuSec, (unsigned long)k, k / cpuSec);
    printf("speedup vs 1 CPU core: %.1fx\n", (double)(n / gpuSec) / (k / cpuSec));
    return 0;
}
