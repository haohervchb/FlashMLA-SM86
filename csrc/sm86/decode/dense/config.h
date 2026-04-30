#pragma once

namespace Config {

// Reduced from 64 to 16 due to RTX 3090's 99KB shared memory limit
static constexpr int BLOCK_SIZE_M = 16;
static constexpr int PAGE_BLOCK_SIZE = 64;

static constexpr int HEAD_DIM_K = 576;
static constexpr int HEAD_DIM_V = 512;

}
