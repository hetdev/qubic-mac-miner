// gpu_miner — GPU solo miner for Qubic (Apple Silicon, Metal).
//
// Mines the bpp9000 standalone walk on the Apple GPU with the bit-exact
// kernel from kernels.metal (bppWalkSimd4) and submits solutions to a Qubic
// node using the exact BROADCAST_MESSAGE protocol of the reference Qiner
// (the node recomputes every score with reference semantics, so the GPU walk
// MUST stay bit-exact — validated by metal_bench and --selftest here).
//
//   ./gpu_miner <NodeIP> <NodePort> <MiningID> <SigningSeed> <MiningSeedHex> \
//               [taskFile] [batchSize] [--selftest|--bench]
//
// - MiningID:      the identity whose solutions are ranked (60-char Qubic id)
// - SigningSeed:   55-char seed that signs solution broadcasts; its identity
//                  must hold QUs for message dissemination and must differ
//                  from the MiningID (same rules as the reference Qiner)
// - MiningSeedHex: 64 hex chars of the epoch mining seed
//
// Like Qiner, mining continues when the node is unreachable (submissions are
// queued and retried); connect to a real node for actual solutions.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "score_bpp9000.h"
#include "keyUtils.h"
#include "portable_rng.h"

// ---------------------------------------------------------------------------
// Reference miner types (same layout/semantics as Qiner.cpp)
// ---------------------------------------------------------------------------
struct RequestResponseHeader
{
private:
    unsigned char _size[3];
    unsigned char _type;
    unsigned int _dejavu;

public:
    inline unsigned int size() { return (*((unsigned int*)_size)) & 0xFFFFFF; }
    inline void setSize(unsigned int size)
    {
        _size[0] = (unsigned char)size;
        _size[1] = (unsigned char)(size >> 8);
        _size[2] = (unsigned char)(size >> 16);
    }
    inline void zeroDejavu() { _dejavu = 0; }
    inline void randomizeDejavu()
    {
        _rdrand32_step(&_dejavu);
        if (!_dejavu) _dejavu = 1;
    }
    inline void setType(const unsigned char type) { _type = type; }
};

#define BROADCAST_MESSAGE 1

typedef struct
{
    unsigned char sourcePublicKey[32];
    unsigned char destinationPublicKey[32];
    unsigned char gammingNonce[32];
} Message;

struct ServerSocket
{
    void closeConnection() { close(serverSocket); }
    bool establishConnection(const char* address, int port)
    {
        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket == -1) return false;
        sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port);
        if (inet_pton(AF_INET, address, &addr.sin_addr) <= 0) { closeConnection(); return false; }
        if (connect(serverSocket, (struct sockaddr*)&addr, sizeof(addr)) < 0) { closeConnection(); return false; }
        return true;
    }
    bool sendData(char* buffer, unsigned int size)
    {
        while (size)
        {
            int n;
            if ((n = (int)send(serverSocket, buffer, size, 0)) <= 0) return false;
            buffer += n; size -= (unsigned int)n;
        }
        return true;
    }
    int serverSocket = -1;
};

// ---------------------------------------------------------------------------
// Scorer + Metal plumbing (mirrors metal_bench.mm / kernels.metal)
// ---------------------------------------------------------------------------
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

struct WalkParamsCPP { uint32_t controlIndex; uint32_t outputIndex; };
static constexpr size_t SEEDS_PER_NONCE = 1000 * 10;
static constexpr uint32_t WALKS_PER_GROUP = 4;

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
    if (!lib) { fprintf(stderr, "MSL error: %s\n", err.localizedDescription.UTF8String); return false; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"bppWalkSimd4"];
    ctx.pipe = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
    if (!ctx.pipe) { fprintf(stderr, "Pipeline error: %s\n", err.localizedDescription.UTF8String); return false; }
    return true;
}

struct GpuInput
{
    std::vector<unsigned long long> seeds;
    uint32_t L, mode;
    unsigned char nonce[32];   // kept for submission on a hit
};

// One batch of walks on the GPU. Returns (error, shift) per input.
static std::vector<std::pair<uint32_t, uint32_t>>
gpuWalks(MetalCtx& ctx, const unsigned char* target,
         const unsigned char* rootLut, const unsigned char* rootStart, const uint16_t* rootNeigh,
         const std::vector<GpuInput>& inputs, uint32_t controlIndex, uint32_t outputIndex)
{
    const NSUInteger n = inputs.size();
    const NSUInteger nPadded = ((n + 3) / 4) * 4;

    id<MTLBuffer> bTarget = [ctx.device newBufferWithBytes:target length:8928 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bLut    = [ctx.device newBufferWithBytes:rootLut   length:1728 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bStart  = [ctx.device newBufferWithBytes:rootStart length:64   options:MTLResourceStorageModeShared];
    id<MTLBuffer> bNeigh  = [ctx.device newBufferWithBytes:rootNeigh length:384  options:MTLResourceStorageModeShared];

    std::vector<unsigned long long> seeds(nPadded * SEEDS_PER_NONCE, 0);
    std::vector<uint32_t> params(nPadded * 2, 0);
    for (NSUInteger i = 0; i < nPadded; ++i) { params[i * 2] = 1; params[i * 2 + 1] = 1; }
    for (NSUInteger i = 0; i < n; ++i)
    {
        memcpy(&seeds[i * SEEDS_PER_NONCE], inputs[i].seeds.data(), SEEDS_PER_NONCE * sizeof(unsigned long long));
        params[i * 2] = inputs[i].L;
        params[i * 2 + 1] = inputs[i].mode;
    }
    id<MTLBuffer> bSeeds  = [ctx.device newBufferWithBytes:seeds.data()  length:seeds.size() * 8 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bParams = [ctx.device newBufferWithBytes:params.data() length:params.size() * 4 options:MTLResourceStorageModeShared];

    WalkParamsCPP wp{ controlIndex, outputIndex };
    id<MTLBuffer> bShared = [ctx.device newBufferWithBytes:&wp length:sizeof(wp) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bState  = [ctx.device newBufferWithLength:WALKS_PER_GROUP * sizeof(TGState4CPP) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bOut    = [ctx.device newBufferWithLength:nPadded * 8 options:MTLResourceStorageModeShared];

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

// ---------------------------------------------------------------------------
// Solution submission (same packet as the reference Qiner)
// ---------------------------------------------------------------------------
struct FoundSolution
{
    std::array<unsigned char, 32> nonce;
    unsigned int score;
};

static std::queue<FoundSolution> foundNonce;
static std::mutex foundNonceLock;

static void hexToByte(const char* hex, uint8_t* byte, const int sizeInByte)
{
    for (int i = 0; i < sizeInByte; i++) sscanf(hex + i * 2, "%2hhx", &byte[i]);
}

static bool submitSolution(const char* nodeIp, int nodePort,
                           const unsigned char* miningSeed,
                           const unsigned char* computorPublicKey,
                           const unsigned char* signingPublicKey,
                           const unsigned char* signingSubseed,
                           const unsigned char* signingPrivateKey,
                           const FoundSolution& sol)
{
    ServerSocket sock;
    if (!sock.establishConnection(nodeIp, nodePort)) return false;

    struct
    {
        RequestResponseHeader header;
        Message message;
        unsigned char solutionMiningSeed[32];
        unsigned char solutionNonce[32];
        unsigned int solutionScore;
        unsigned char signature[64];
    } packet;

    packet.header.setSize(sizeof(packet));
    packet.header.zeroDejavu();
    packet.header.setType(BROADCAST_MESSAGE);

    memcpy(packet.message.sourcePublicKey, signingPublicKey, 32);
    memcpy(packet.message.destinationPublicKey, computorPublicKey, 32);

    unsigned char sharedKeyAndGammingNonce[64];
    memset(sharedKeyAndGammingNonce, 0, 32);
    if (memcmp(computorPublicKey, signingPublicKey, 32) == 0)
        getSharedKey(signingPrivateKey, computorPublicKey, sharedKeyAndGammingNonce);

    unsigned char gammingKey[32];
    do
    {
        _rdrand64_step((unsigned long long*)&packet.message.gammingNonce[0]);
        _rdrand64_step((unsigned long long*)&packet.message.gammingNonce[8]);
        _rdrand64_step((unsigned long long*)&packet.message.gammingNonce[16]);
        _rdrand64_step((unsigned long long*)&packet.message.gammingNonce[24]);
        memcpy(&sharedKeyAndGammingNonce[32], packet.message.gammingNonce, 32);
        KangarooTwelve(sharedKeyAndGammingNonce, 64, gammingKey, 32);
    } while (gammingKey[0]);

    unsigned char gamma[32 + 32 + 4];
    KangarooTwelve(gammingKey, sizeof(gammingKey), gamma, sizeof(gamma));
    for (unsigned int i = 0; i < 32; i++)
    {
        packet.solutionMiningSeed[i] = miningSeed[i] ^ gamma[i];
        packet.solutionNonce[i] = sol.nonce[i] ^ gamma[32 + i];
    }
    for (unsigned int i = 0; i < 4; i++)
        ((unsigned char*)&packet.solutionScore)[i] = ((const unsigned char*)&sol.score)[i] ^ gamma[64 + i];

    unsigned char digest[32] = {0};
    unsigned char signature[64] = {0};
    KangarooTwelve((unsigned char*)&packet + sizeof(RequestResponseHeader),
                   sizeof(packet) - sizeof(RequestResponseHeader) - 64, digest, 32);
    sign(signingSubseed, signingPublicKey, digest, signature);
    memcpy(packet.signature, signature, 64);

    const bool ok = sock.sendData((char*)&packet, packet.header.size());
    sock.closeConnection();
    return ok;
}

// ---------------------------------------------------------------------------
// Mining loop
// ---------------------------------------------------------------------------
static std::atomic<bool> g_stop(false);
static std::atomic<unsigned long long> g_totalNonces(0);
static std::atomic<unsigned int> g_solutions(0);

static void ctrlCHandler(int) { g_stop = true; }

static void canonicalNonce(unsigned char* nonce)
{
    _rdrand64_step((unsigned long long*)&nonce[0]);
    _rdrand64_step((unsigned long long*)&nonce[8]);
    _rdrand64_step((unsigned long long*)&nonce[16]);
    _rdrand64_step((unsigned long long*)&nonce[24]);
    nonce[0] = (unsigned char)AlgoType::Bpp9000;
    const unsigned char L = (unsigned char)((nonce[1] % score_bpp9000::MAX_CHANGES_PER_STEP) + 1);
    const unsigned char mode = (unsigned char)(((nonce[1] >> 4) % 3) + 1);
    nonce[1] = (unsigned char)(L | (mode << 4));
    nonce[2] = 0;
}

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

int main(int argc, char** argv)
{
    if (argc < 6)
    {
        printf("Usage: gpu_miner <NodeIP> <NodePort> <MiningID> <SigningSeed> <MiningSeedHex> [taskFile] [batchSize] [--selftest|--bench]\n");
        return 1;
    }
    const char* nodeIp = argv[1];
    const int nodePort = atoi(argv[2]);
    const char* miningID = argv[3];
    const char* signingSeed = argv[4];
    std::string taskPath = "../qiner-mac/test/data/bpp9000.task";
    std::string kernelPath = "kernels.metal";
    NSUInteger batchSize = 512;
    bool selftest = false, benchOnly = false;
    {
        std::string miningSeedHex = argv[5];
        int positional = 0;
        for (int i = 6; i < argc; ++i)
        {
            if (strcmp(argv[i], "--selftest") == 0) selftest = true;
            else if (strcmp(argv[i], "--bench") == 0) benchOnly = true;
            else if (positional == 0) { taskPath = argv[i]; positional++; }
            else if (positional == 1) { batchSize = (NSUInteger)atoi(argv[i]); positional++; }
            else kernelPath = argv[i];
        }
    }
    unsigned char miningSeed[32];
    if (!parseHex32(argv[5], miningSeed)) { fprintf(stderr, "bad MiningSeedHex\n"); return 1; }
    if (batchSize < 4 || batchSize > 65536)
    {
        fprintf(stderr, "batchSize must be 4..65536 (got %lu)\n", (unsigned long)batchSize);
        return 1;
    }

    // ---- identities (same derivation as Qiner) ----
    unsigned char computorPublicKey[32];
    getPublicKeyFromIdentity(miningID, computorPublicKey);
    unsigned char signingPrivateKey[32], signingSubseed[32], signingPublicKey[32];
    getSubseedFromSeed((const unsigned char*)signingSeed, signingSubseed);
    getPrivateKeyFromSubSeed(signingSubseed, signingPrivateKey);
    getPublicKeyFromPrivateKey(signingPrivateKey, signingPublicKey);

    signal(SIGINT, ctrlCHandler);

    MetalCtx ctx;
    if (!initMetal(ctx, kernelPath)) return 1;
    printf("Metal device: %s | kernel bppWalkSimd4 | batch %lu\n", ctx.device.name.UTF8String, (unsigned long)batchSize);

    // ---- optional selftest: golden nonce must reproduce error=4811 ----
    if (selftest)
    {
        unsigned char gSeed[32], gPub[32], gNonce[32];
        parseHex32("8b12add89bc264e01038b61b784e0778a03cd3025e1faeedda6598f197ceccc0", gSeed);
        parseHex32("039cc1f1560aa96daa994a2b296f22d7f2fc9503ce95321d0b8193079e5f93dc", gPub);
        parseHex32("bee1e55fc2c967601827dd80f09a6b0a47ceab37b920b57efbd23c8fdc49e38f", gNonce);
        std::vector<unsigned char> pool(POOL_VEC_PADDING_SIZE);
        generateRandom2Pool(gSeed, pool.data());
        ProdMiner m;
        m.setPool(pool.data());
        m.deriveControlOutput(gSeed);
        if (!m.loadTaskData(taskPath.c_str())) { fprintf(stderr, "selftest: task load failed\n"); return 1; }

        GpuInput in;
        unsigned char target[8928];
        for (unsigned long long i = 0; i < score_bpp9000::SEQUENCE_LENGTH; ++i) target[i] = m.outputs[i][0];
        m.deriveRootMaterial(gPub);
        m.deriveMutationSeeds(gPub, gNonce, nullptr);
        in.seeds.assign(m.mutationSeed, m.mutationSeed + SEEDS_PER_NONCE);
        in.L = gNonce[1] & 0x0F;
        in.mode = (gNonce[1] >> 4) & 0x03;
        std::vector<unsigned char> lut(1728), start(64);
        std::vector<uint16_t> neigh(192);
        for (int i = 0; i < 1728; ++i) lut[i] = (unsigned char)(m.rootMaterial.lut[i] % 3);
        for (int n = 0; n < 64; ++n)   start[n] = (unsigned char)(m.rootMaterial.start[n] % 3);
        for (int i = 0; i < 192; ++i)  neigh[i] = (uint16_t)(m.rootMaterial.wire[i] % 64);
        auto r = gpuWalks(ctx, target, lut.data(), start.data(), neigh.data(), { in }, m.controlIndex, m.outputIndex)[0];
        printf("selftest: GPU error=%u shift=%u (expect 4811) -> %s\n", r.first, r.second,
               r.first == 4811 ? "PASS" : "FAIL");
        if (r.first != 4811) return 1;
    }

    // ---- shared random2 pool + task ----
    std::vector<unsigned char> pool(POOL_VEC_PADDING_SIZE);
    generateRandom2Pool(miningSeed, pool.data());
    ProdMiner m;
    m.setPool(pool.data());
    m.deriveControlOutput(miningSeed);
    if (!m.loadTaskData(taskPath.c_str())) { fprintf(stderr, "Failed to load task %s\n", taskPath.c_str()); return 1; }

    unsigned char target[8928];
    for (unsigned long long i = 0; i < score_bpp9000::SEQUENCE_LENGTH; ++i) target[i] = m.outputs[i][0];

    m.deriveRootMaterial(computorPublicKey);
    std::vector<unsigned char> rootLut(1728), rootStart(64);
    std::vector<uint16_t> rootNeigh(192);
    for (int i = 0; i < 1728; ++i) rootLut[i] = (unsigned char)(m.rootMaterial.lut[i] % 3);
    for (int n = 0; n < 64; ++n)   rootStart[n] = (unsigned char)(m.rootMaterial.start[n] % 3);
    for (int i = 0; i < 192; ++i)  rootNeigh[i] = (uint16_t)(m.rootMaterial.wire[i] % 64);

    // ---- submission thread (same retry semantics as Qiner) ----
    std::thread sender([&]()
    {
        while (!g_stop)
        {
            FoundSolution sol;
            bool have = false;
            {
                std::lock_guard<std::mutex> g(foundNonceLock);
                if (!foundNonce.empty()) { sol = foundNonce.front(); have = true; }
            }
            if (have)
            {
                if (submitSolution(nodeIp, nodePort, miningSeed, computorPublicKey,
                                   signingPublicKey, signingSubseed, signingPrivateKey, sol))
                {
                    std::lock_guard<std::mutex> g(foundNonceLock);
                    if (!foundNonce.empty()) foundNonce.pop();
                    printf("[submit] solution score=%u sent to %s:%d\n", sol.score, nodeIp, nodePort);
                }
                else
                {
                    printf("[submit] node %s:%d unreachable, retrying...\n", nodeIp, nodePort);
                }
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });

    // ---- mining loop: stage on CPU, walk on GPU, gate solutions ----
    printf("mining (GPU walk, reference-exact) | Ctrl+C to stop\n");
    auto statsStart = std::chrono::steady_clock::now();
    unsigned long long statsBase = 0;
    while (!g_stop)
    {
        // stage one batch (inputs derived by the reference code)
        std::vector<GpuInput> inputs(batchSize);
        for (NSUInteger i = 0; i < batchSize; ++i)
        {
            canonicalNonce(inputs[i].nonce);
            m.deriveMutationSeeds(computorPublicKey, inputs[i].nonce, nullptr);
            inputs[i].seeds.assign(m.mutationSeed, m.mutationSeed + SEEDS_PER_NONCE);
            inputs[i].L = inputs[i].nonce[1] & 0x0F;
            inputs[i].mode = (inputs[i].nonce[1] >> 4) & 0x03;
        }

        auto ratings = gpuWalks(ctx, target, rootLut.data(), rootStart.data(), rootNeigh.data(),
                                inputs, m.controlIndex, m.outputIndex);

        for (NSUInteger i = 0; i < batchSize; ++i)
        {
            const uint32_t error = ratings[i].first;
            const uint32_t shift = ratings[i].second;
            const bool valid = error != 0xFFFFFFFFu;
            const bool clears = (shift > 0) || (error <= score_bpp9000::SOLUTION_THRESHOLD);
            if (valid && clears)
            {
                FoundSolution sol;
                memcpy(sol.nonce.data(), inputs[i].nonce, 32);
                sol.score = error;
                {
                    std::lock_guard<std::mutex> g(foundNonceLock);
                    foundNonce.push(sol);
                }
                g_solutions++;
                printf("[HIT] nonce solution error=%u shift=%u -> queued for submission\n", error, shift);
            }
        }
        g_totalNonces += batchSize;

        // stats every ~5 s
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(now - statsStart).count();
        if (dt >= 5.0)
        {
            const unsigned long long total = g_totalNonces.load();
            printf("| %s | %.1f nonces/s | total %llu | solutions %u |\n",
                   benchOnly ? "bench" : "mining", (total - statsBase) / dt, total, g_solutions.load());
            statsStart = now;
            statsBase = total;
        }
        if (benchOnly && g_totalNonces.load() >= batchSize * 4) break;   // ~4 batches then exit
    }

    g_stop = true;
    if (sender.joinable()) sender.join();
    printf("stopped | total nonces %llu | solutions %u\n", g_totalNonces.load(), g_solutions.load());
    return 0;
}
