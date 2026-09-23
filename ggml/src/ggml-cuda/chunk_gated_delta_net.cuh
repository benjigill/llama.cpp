#pragma once

#include "common.cuh"
#include "gated_delta_net.cuh"
#include "ggml.h"

void ggml_cuda_op_gated_delta_net_chunked(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                          const ggml_cuda_gated_delta_net_fused_cache * cache);

// raw launch for one span of tokens: attention rows go to out, the final state to state_dst
// dst only locates the scratch, so the span may be shorter than dst's token count
void ggml_cuda_op_gated_delta_net_chunked_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * dst,
                                               float * out, float * state_dst,
                                               int B, int T, int H, int num_k_heads, int k_dim, int v_dim, int num_chunks,
                                               const float * q_in, const float * k_in, const float * v_in,
                                               const float * g_in, const float * b_in, const float * s_d,
                                               float scale, long long v_tok_stride, cudaStream_t stream);

// Chunked-GDN scratch, carved out of the tail of dst's own allocation. Two reasons:
//  1. The graph allocator sizes every tensor through ggml_backend_buft_get_alloc_size, so scratch
//     that lives in dst's buffer is visible to llama_params_fit (--fit). A context-owned cudaMalloc
//     (or the old ggml_cuda_pool_alloc) is invisible to that projection, which made --fit
//     under-estimate VRAM by the full scratch size.
//  2. The address is a fixed offset from dst->data, so it is stable across CUDA-graph capture and
//     replay without needing a separate persistent allocation to be pre-sized before capture.
struct ggml_cuda_gdn_chunked_scratch {
    float *   v_corr;
    float *   k_cumdecay;
    float *   g_cum;
    float *   qk;
    uintptr_t end;
};

// Pure function of the tensor graph, so allocation time and execution time cannot disagree.
ggml_cuda_gdn_chunked_scratch ggml_cuda_gdn_get_chunked_scratch(const ggml_tensor * dst);

// ggml_nbytes(dst) plus the scratch above, or just ggml_nbytes(dst) when the shape is ineligible.
size_t ggml_cuda_gdn_get_alloc_size(const ggml_tensor * dst);
