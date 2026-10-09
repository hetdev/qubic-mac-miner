// bpp9000 walk on Metal — clean-room port of the reference scorer in
// qiner-mac/src/score_bpp9000.h (computeScoreFromCurrent / advanceShift /
// score / mutate with K = 0 standalone semantics).
//
// One GPU thread = one complete nonce walk. The math is pure integer/trit
// arithmetic, so the output MUST be bit-identical to the CPU scorer; the host
// (metal_bench.mm) validates exactly that against the reference C++ code.
//
// Per-thread state lives in thread-private memory (threadgroup size 1, so the
// ~7.7 KB of private state fits). Inputs (root material, mutation seeds,
// target sequence) are computed on the CPU by the reference implementation
// (K12/random2 stay CPU-side; the walk is the hot loop being accelerated).

#include <metal_stdlib>
using namespace metal;

// Production constants (must match score_bpp9000.h)
constant uint POPULATION     = 64;
constant uint NEIGHBORS      = 3;
constant uint LUT_SIZE       = 27;
constant uint LINKS          = NEIGHBORS * POPULATION;      // 192
constant uint LUT_TOTAL      = LUT_SIZE * POPULATION;       // 1728
constant uint WINDOW_WIDTH   = 24 * 365;                    // 8760
constant uint SHIFT_CAP      = 24 * 7;                      // 168
constant uint ADVANCE_THR    = WINDOW_WIDTH / 3;            // 2920
constant ulong MAX_TICKS     = 100000;
constant uint MUTATIONS      = 1000;
constant uint MAX_L          = 10;
constant uint INFINITE_ERROR = 0xFFFFFFFFu;

struct WalkParams
{
    uint controlIndex;
    uint outputIndex;
};

struct WalkState
{
    uchar  curLut[LUT_TOTAL];
    uchar  curInit[POPULATION];
    ushort curNeigh[LINKS];
    uchar  prevLut[LUT_TOTAL];
    uchar  prevInit[POPULATION];
    ushort prevNeigh[LINKS];
    uchar  bestLut[LUT_TOTAL];
    uchar  bestInit[POPULATION];
    ushort bestNeigh[LINKS];
};

static inline bool ratingValid(uint2 r) { return r.x != INFINITE_ERROR; }

// isNotWorseThan(other) = !other.isBetterThan(this)
static inline bool isNotWorse(uint2 self, uint2 other)
{
    if (other.y != self.y) return !(other.y > self.y);
    return !(other.x < self.x);
}

// score(): autonomous rollout; grades [shift, shift + WINDOW_WIDTH).
// Returns the frame error or INFINITE_ERROR on tick timeout.
static uint scoreFrame(device WalkState& st, constant uchar* target,
                       ulong shift, uint controlIndex, uint outputIndex)
{
    uchar neuronOut[POPULATION];
    uchar neuronPrev[POPULATION];
    for (uint n = 0; n < POPULATION; ++n) neuronOut[n] = st.curInit[n];

    uint failures = 0;
    ulong counter = 0;
    ulong ticks = 0;
    while (counter < shift + WINDOW_WIDTH)
    {
        if (++ticks >= MAX_TICKS) return INFINITE_ERROR;

        for (uint n = 0; n < POPULATION; ++n) neuronPrev[n] = neuronOut[n];
        for (uint n = 0; n < POPULATION; ++n)
        {
            const uchar t0 = neuronPrev[st.curNeigh[n * NEIGHBORS + 0]];
            const uchar t1 = neuronPrev[st.curNeigh[n * NEIGHBORS + 1]];
            const uchar t2 = neuronPrev[st.curNeigh[n * NEIGHBORS + 2]];
            neuronOut[n] = st.curLut[n * LUT_SIZE + (uint)(t0 + 3 * t1 + 9 * t2)];
        }

        if (neuronOut[controlIndex] != 2)   // TRIT_UNKNOWN
        {
            if (counter >= shift && neuronOut[outputIndex] != target[counter])
                ++failures;
            ++counter;
        }
    }
    return failures;
}

// advanceShift(): slide the frame while the network masters it. Returns
// {frameError, shift} exactly like Rating{frameError, (unsigned)shift}.
static uint2 advanceShift(device WalkState& st, constant uchar* target,
                          thread ulong& shift, uint controlIndex, uint outputIndex)
{
    uint frameErr;
    for (;;)
    {
        frameErr = scoreFrame(st, target, shift, controlIndex, outputIndex);
        if (frameErr > ADVANCE_THR) break;   // cannot master this frame
        if (shift == SHIFT_CAP) break;       // the cap
        ++shift;
    }
    return uint2(frameErr, (uint)shift);
}

static void mutate(device WalkState& st, uint mode, ulong v)
{
    if (mode == 1)          // start state
    {
        const uchar d = (uchar)(v & 1ULL);
        const uint nn = (uint)((v >> 1) % POPULATION);
        st.curInit[nn] = (uchar)((st.curInit[nn] + 1 + d) % 3);
    }
    else if (mode == 2)     // wiring
    {
        const uint slot = (uint)(v % LINKS);
        uint tgt = (uint)((v / LINKS) % POPULATION);
        const uint prev = st.curNeigh[slot];
        while (tgt == prev) tgt = (tgt + 1) % POPULATION;
        st.curNeigh[slot] = (ushort)tgt;
    }
    else                    // LUT
    {
        const uchar d = (uchar)(v & 1ULL);
        const uint idx = (uint)((v >> 1) % LUT_TOTAL);
        st.curLut[idx] = (uchar)((st.curLut[idx] + 1 + d) % 3);
    }
}

kernel void bppWalk(
    constant uchar*  target      [[buffer(0)]],   // SEQUENCE_LENGTH target trits
    constant uchar*  rootLut     [[buffer(1)]],   // 1728 (already % 3)
    constant uchar*  rootStart   [[buffer(2)]],   // 64   (already % 3)
    constant ushort* rootNeigh   [[buffer(3)]],   // 192  (already % POPULATION)
    constant ulong*  seeds       [[buffer(4)]],   // MUTATIONS*MAX_L per nonce
    constant uint*   nonceParams [[buffer(5)]],   // 2 per nonce: (L, mode)
    constant WalkParams& shared  [[buffer(6)]],
    device WalkState* states     [[buffer(7)]],   // one walk state per nonce
    device uint2*    outRating   [[buffer(8)]],   // (error, shift) per nonce
    uint gid [[thread_position_in_grid]])
{
    device WalkState& st = states[gid];

    const uint L    = nonceParams[gid * 2 + 0];
    const uint mode = nonceParams[gid * 2 + 1];
    constant ulong* mySeeds = seeds + (ulong)gid * (MUTATIONS * MAX_L);
    const uint controlIndex = shared.controlIndex;
    const uint outputIndex  = shared.outputIndex;

    // applyRootMaterial()
    for (uint i = 0; i < LUT_TOTAL; ++i)  st.curLut[i]  = rootLut[i];
    for (uint n = 0; n < POPULATION; ++n) st.curInit[n] = rootStart[n];
    for (uint i = 0; i < LINKS; ++i)      st.curNeigh[i] = rootNeigh[i];

    ulong shift = 0;

    // computeScoreFromCurrent(L, K = 0, mode)
    uint2 cur = advanceShift(st, target, shift, controlIndex, outputIndex);
    uint2 best = cur;
    for (uint i = 0; i < LUT_TOTAL; ++i)  st.bestLut[i]  = st.curLut[i];
    for (uint n = 0; n < POPULATION; ++n) st.bestInit[n] = st.curInit[n];
    for (uint i = 0; i < LINKS; ++i)      st.bestNeigh[i] = st.curNeigh[i];

    for (uint s = 0; s < MUTATIONS; ++s)
    {
        // snapshotPrev()
        for (uint i = 0; i < LUT_TOTAL; ++i)  st.prevLut[i]  = st.curLut[i];
        for (uint n = 0; n < POPULATION; ++n) st.prevInit[n] = st.curInit[n];
        for (uint i = 0; i < LINKS; ++i)      st.prevNeigh[i] = st.curNeigh[i];
        const ulong prevShift = shift;

        for (uint i = 0; i < L; ++i)
            mutate(st, mode, mySeeds[s * MAX_L + i]);

        const uint2 r = advanceShift(st, target, shift, controlIndex, outputIndex);

        // K = 0: exploit branch only (keep-if-not-worse, one-step rollback)
        if (ratingValid(r) && isNotWorse(r, cur))
        {
            cur = r;
        }
        else
        {
            for (uint i = 0; i < LUT_TOTAL; ++i)  st.curLut[i]  = st.prevLut[i];
            for (uint n = 0; n < POPULATION; ++n) st.curInit[n] = st.prevInit[n];
            for (uint i = 0; i < LINKS; ++i)      st.curNeigh[i] = st.prevNeigh[i];
            shift = prevShift;
        }

        if (ratingValid(cur) && isNotWorse(cur, best))
        {
            best = cur;
            for (uint i = 0; i < LUT_TOTAL; ++i)  st.bestLut[i]  = st.curLut[i];
            for (uint n = 0; n < POPULATION; ++n) st.bestInit[n] = st.curInit[n];
            for (uint i = 0; i < LINKS; ++i)      st.bestNeigh[i] = st.curNeigh[i];
        }
    }

    outRating[gid] = best;
}

// ============================================================================
// v2: SIMD-group walk — one threadgroup (32 lanes, one simdgroup) per nonce.
//
// The ONLY data-parallel part of the reference walk that can be widened
// WITHOUT changing the per-nonce trajectory (the node recomputes every nonce
// with reference semantics, so the walk must stay bit-exact) is the neuron
// step inside score(): all 64 neurons of a tick read the previous state and
// write the next one independently. Here lanes cooperate on those 64 neurons
// (2 per lane) and the walk state lives in threadgroup memory (ping-pong
// neuron buffers instead of the per-tick memcpy).
// ============================================================================

struct TGState
{
    uchar  curLut[LUT_TOTAL];
    uchar  curInit[POPULATION];
    ushort curNeigh[LINKS];
    uchar  prevLut[LUT_TOTAL];
    uchar  prevInit[POPULATION];
    ushort prevNeigh[LINKS];
    uchar  bestLut[LUT_TOTAL];
    uchar  bestInit[POPULATION];
    ushort bestNeigh[LINKS];
    uchar  neuronA[POPULATION];
    uchar  neuronB[POPULATION];
    uint   counter;
    uint   failures;
    ulong  ticks;
    ulong  shift;
    uint   pendingShift;
};

static inline void tgCopyLut(threadgroup uchar* dst, threadgroup uchar* src, uint lane)
{
    for (uint i = lane; i < LUT_TOTAL; i += 32) dst[i] = src[i];
}
static inline void tgCopyInit(threadgroup uchar* dst, threadgroup uchar* src, uint lane)
{
    for (uint n = lane; n < POPULATION; n += 32) dst[n] = src[n];
}
static inline void tgCopyNeigh(threadgroup ushort* dst, threadgroup ushort* src, uint lane)
{
    for (uint i = lane; i < LINKS; i += 32) dst[i] = src[i];
}

kernel void bppWalkSimd(
    constant uchar*  target      [[buffer(0)]],
    constant uchar*  rootLut     [[buffer(1)]],
    constant uchar*  rootStart   [[buffer(2)]],
    constant ushort* rootNeigh   [[buffer(3)]],
    constant ulong*  seeds       [[buffer(4)]],
    constant uint*   nonceParams [[buffer(5)]],
    constant WalkParams& shared  [[buffer(6)]],
    device uint2*    outRating   [[buffer(8)]],
    threadgroup uchar* tgRaw     [[threadgroup(0)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    threadgroup TGState& st = *(threadgroup TGState*)tgRaw;

    const uint L    = nonceParams[gid * 2 + 0];
    const uint mode = nonceParams[gid * 2 + 1];
    constant ulong* mySeeds = seeds + (ulong)gid * (MUTATIONS * MAX_L);
    const uint controlIndex = shared.controlIndex;
    const uint outputIndex  = shared.outputIndex;

    // applyRootMaterial() — parallel copy
    for (uint i = lane; i < LUT_TOTAL; i += 32) st.curLut[i] = rootLut[i];
    for (uint n = lane; n < POPULATION; n += 32) st.curInit[n] = rootStart[n];
    for (uint i = lane; i < LINKS; i += 32) st.curNeigh[i] = rootNeigh[i];
    if (lane == 0) { st.counter = 0; st.failures = 0; st.ticks = 0; st.shift = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- scoreFrame: 64 neurons per tick across 32 lanes (ping-pong) ----
    #define SCORE_FRAME() do {                                             \
        for (uint n = lane; n < POPULATION; n += 32) st.neuronA[n] = st.curInit[n]; \
        if (lane == 0) { st.counter = 0; st.failures = 0; st.ticks = 0; }   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
        threadgroup uchar* src = st.neuronA;                                \
        threadgroup uchar* dst = st.neuronB;                                \
        while (st.counter < st.shift + WINDOW_WIDTH)                        \
        {                                                                   \
            if (lane == 0)                                                  \
            {                                                               \
                ++st.ticks;                                                 \
                if (st.ticks >= MAX_TICKS) st.failures = INFINITE_ERROR;    \
            }                                                               \
            threadgroup_barrier(mem_flags::mem_threadgroup);                \
            if (st.failures == INFINITE_ERROR) break;                       \
            for (uint k = 0; k < POPULATION / 32; ++k)                      \
            {                                                               \
                const uint n = lane + 32 * k;                               \
                const uchar t0 = src[st.curNeigh[n * NEIGHBORS + 0]];       \
                const uchar t1 = src[st.curNeigh[n * NEIGHBORS + 1]];       \
                const uchar t2 = src[st.curNeigh[n * NEIGHBORS + 2]];       \
                dst[n] = st.curLut[n * LUT_SIZE + (uint)(t0 + 3 * t1 + 9 * t2)]; \
            }                                                               \
            threadgroup_barrier(mem_flags::mem_threadgroup);                \
            if (dst[controlIndex] != 2)                                     \
            {                                                               \
                if (lane == 0)                                              \
                {                                                           \
                    if (st.counter >= st.shift &&                           \
                        dst[outputIndex] != target[st.counter])             \
                        ++st.failures;                                      \
                    ++st.counter;                                           \
                }                                                           \
            }                                                               \
            threadgroup_barrier(mem_flags::mem_threadgroup);                \
            threadgroup uchar* tmp = src; src = dst; dst = tmp;             \
        }                                                                   \
    } while (0)

    // ---- advanceShift ----
    #define TG_ADVANCE() do {                                              \
        for (;;)                                                           \
        {                                                                  \
            SCORE_FRAME();                                                 \
            if (st.failures == INFINITE_ERROR || st.failures > ADVANCE_THR) break; \
            if (st.shift == SHIFT_CAP) break;                              \
            if (lane == 0) ++st.shift;                                     \
            threadgroup_barrier(mem_flags::mem_threadgroup);               \
        }                                                                  \
    } while (0)

    // ---- walk: computeScoreFromCurrent(L, 0, mode) ----
    TG_ADVANCE();
    uint2 cur  = uint2(st.failures, (uint)st.shift);
    uint2 best = cur;
    tgCopyLut(st.bestLut, st.curLut, lane);
    tgCopyInit(st.bestInit, st.curInit, lane);
    tgCopyNeigh(st.bestNeigh, st.curNeigh, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < MUTATIONS; ++s)
    {
        tgCopyLut(st.prevLut, st.curLut, lane);
        tgCopyInit(st.prevInit, st.curInit, lane);
        tgCopyNeigh(st.prevNeigh, st.curNeigh, lane);
        if (lane == 0)
        {
            const ulong prevShift = st.shift;
            // L sequential mutations (same seeds, same order as the reference)
            for (uint i = 0; i < L; ++i)
            {
                const ulong v = mySeeds[s * MAX_L + i];
                if (mode == 1)
                {
                    const uchar d = (uchar)(v & 1ULL);
                    const uint nn = (uint)((v >> 1) % POPULATION);
                    st.curInit[nn] = (uchar)((st.curInit[nn] + 1 + d) % 3);
                }
                else if (mode == 2)
                {
                    const uint slot = (uint)(v % LINKS);
                    uint tgt = (uint)((v / LINKS) % POPULATION);
                    const uint prev = st.curNeigh[slot];
                    while (tgt == prev) tgt = (tgt + 1) % POPULATION;
                    st.curNeigh[slot] = (ushort)tgt;
                }
                else
                {
                    const uchar d = (uchar)(v & 1ULL);
                    const uint idx = (uint)((v >> 1) % LUT_TOTAL);
                    st.curLut[idx] = (uchar)((st.curLut[idx] + 1 + d) % 3);
                }
            }
            st.pendingShift = (uint)prevShift;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        TG_ADVANCE();
        const uint2 r = uint2(st.failures, (uint)st.shift);

        if (lane == 0)
        {
            const uint2 curLocal = cur;
            if (ratingValid(r) && isNotWorse(r, curLocal))
            {
                cur = r;
            }
            else
            {
                // rollback + shift restore
                for (uint i = 0; i < LUT_TOTAL; ++i)  st.curLut[i]  = st.prevLut[i];
                for (uint n = 0; n < POPULATION; ++n) st.curInit[n] = st.prevInit[n];
                for (uint i = 0; i < LINKS; ++i)      st.curNeigh[i] = st.prevNeigh[i];
                st.shift = st.pendingShift;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0)
        {
            const uint2 bestLocal = best;
            if (ratingValid(cur) && isNotWorse(cur, bestLocal))
            {
                best = cur;
                for (uint i = 0; i < LUT_TOTAL; ++i)  st.bestLut[i]  = st.curLut[i];
                for (uint n = 0; n < POPULATION; ++n) st.bestInit[n] = st.curInit[n];
                for (uint i = 0; i < LINKS; ++i)      st.bestNeigh[i] = st.curNeigh[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) outRating[gid] = best;
}
