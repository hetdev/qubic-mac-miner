#pragma once

#include <cstdint>
#include "portable_rng.h"
#include <vector>

#include "task_file.h"

// Generate synthetic data for testing
namespace bpp9000_synth
{

inline unsigned long long rdrand64()
{
    unsigned long long v = 0;
    while (!_rdrand64_step((unsigned long long*)&v))
    {
    }
    return v;
}

inline void fillRandom(unsigned char* out, size_t n)
{
    size_t i = 0;
    while (i < n)
    {
        unsigned long long v = rdrand64();
        for (int b = 0; b < 8 && i < n; ++b, ++i)
        {
            out[i] = (unsigned char)(v & 0xFFULL);
            v >>= 8;
        }
    }
}

// Build a synthetic task for config C by random-filling the two blocks and fixing invalid values
template<typename C>
void buildSyntheticTaskBlocks(std::vector<unsigned char>& topoBlock,
                              std::vector<unsigned char>& dataBlock)
{
    const uint32_t N = (uint32_t)C::numberOfInputNeurons;
    const uint32_t M = (uint32_t)C::numberOfOutputNeurons;
    const uint32_t K = (uint32_t)C::numberOfNeighbors;
    const uint32_t P = (uint32_t)C::populationThreshold;
    const uint64_t T = (uint64_t)C::sequenceLength;

    // Random fill training and topology block
    std::vector<uint32_t> inputIdx(N);
    std::vector<uint32_t> outputIdx(M);
    std::vector<uint32_t> neighborIdx((size_t)P * K);
    uint32_t signalIdx = 0;
    fillRandom((unsigned char*)inputIdx.data(), (size_t)N * sizeof(uint32_t));
    fillRandom((unsigned char*)outputIdx.data(), (size_t)M * sizeof(uint32_t));
    fillRandom((unsigned char*)&signalIdx, sizeof(uint32_t));
    fillRandom((unsigned char*)neighborIdx.data(), (size_t)P * K * sizeof(uint32_t));

    // Make data valid from random value
    // Placements must be in range and mutually distinct; neighbours in range and never the owning
    // neuron (no self-reference).
    std::vector<bool> used(P, false);
    for (uint32_t i = 0; i < N; ++i)
    {
        inputIdx[i] %= P;
        while (used[inputIdx[i]])
        {
            inputIdx[i] = (inputIdx[i] + 1) % P;
        }
        used[inputIdx[i]] = true;
    }
    for (uint32_t j = 0; j < M; ++j)
    {
        outputIdx[j] %= P;
        while (used[outputIdx[j]])
        {
            outputIdx[j] = (outputIdx[j] + 1) % P;
        }
        used[outputIdx[j]] = true;
    }
    signalIdx %= P;
    while (used[signalIdx])
    {
        signalIdx = (signalIdx + 1) % P;
    }
    used[signalIdx] = true;
    for (size_t i = 0; i < (size_t)P * K; ++i)
    {
        const uint32_t owner = (uint32_t)(i / K);
        neighborIdx[i] %= P;
        if (neighborIdx[i] == owner)
        {
            neighborIdx[i] = (neighborIdx[i] + 1) % P;
        }
    }

    topoBlock.resize((size_t)task_file::topologyBytes(N, M, P, K));
    task_file::serializeTopologyBlock(N, M, P, K, inputIdx.data(), outputIdx.data(),
                                      signalIdx, neighborIdx.data(), topoBlock.data());

    dataBlock.resize((size_t)task_file::dataBytes(N, M, T));
    fillRandom(dataBlock.data(), dataBlock.size());
    for (size_t i = 0; i < dataBlock.size(); ++i)
    {
        if (dataBlock[i] >= task_file::BYTE_VALUE_LIMIT)
        {
            dataBlock[i] = (unsigned char)(dataBlock[i] % task_file::BYTE_VALUE_LIMIT);
        }
    }
}

}
