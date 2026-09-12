#pragma once

// Verified against both:
//   1. TinyLlama's official GitHub README at commit
//      bf122247c486b6b897050e98cbb7bedae8eeba73.
//   2. TinyLlama/TinyLlama-1.1B-Chat-v1.0 config.json on Hugging Face,
//      whose model-repository revision was 26d45f2 when inspected.
// Source snapshot and URLs live in ../reference/TINYLLAMA_CONFIG_SOURCE.md.

namespace tinyllama {

inline constexpr int kHiddenSize = 2048;
inline constexpr int kIntermediateSize = 5632;
inline constexpr int kNumHiddenLayers = 22;
inline constexpr int kNumAttentionHeads = 32;
inline constexpr int kNumKeyValueHeads = 4;
inline constexpr int kMaxPositionEmbeddings = 2048;
inline constexpr int kHeadDim = kHiddenSize / kNumAttentionHeads;

static_assert(kHeadDim == 64, "TinyLlama head dimension must be 64");
static_assert(kHiddenSize % kNumAttentionHeads == 0,
              "hidden size must be divisible by attention heads");
static_assert(kNumAttentionHeads % kNumKeyValueHeads == 0,
              "query heads must divide evenly among KV groups");

}  // namespace tinyllama
