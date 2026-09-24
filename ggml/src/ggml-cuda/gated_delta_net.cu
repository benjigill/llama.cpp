#include "gated_delta_net.cuh"
#include "chunk_gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

constexpr int gdn_d_v_per_warp = 4;
constexpr int gdn_num_warps    = 4;
constexpr int gdn_block_dv     = gdn_num_warps * gdn_d_v_per_warp;  // 16

// row-per-warp recurrent kernel (upstream #22587): each warp owns gdn_d_v_per_warp rows of the
// transposed state and its lanes shard the QK axis, so k/q are loaded once per warp per token
// and the per-row dot products are plain warp reductions.
template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__(ggml_cuda_get_physical_warp_size() * gdn_num_warps, 2)
gated_delta_net_cuda(const float * __restrict__ q,
                                     const float * __restrict__ k,
                                     const float * __restrict__ v,
                                     const float * __restrict__ g,
                                     const float * __restrict__ beta,
                                     const float *              curr_state,
                                     float *       __restrict__ dst,
                                     float *                    state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size();
    constexpr int active_lanes  = (S_v < warp_size) ? S_v : warp_size;
    constexpr int d_qk_per_lane = S_v / active_lanes;
    static_assert(d_qk_per_lane >= 1, "S_v must >= 1");
    static_assert(S_v % active_lanes == 0, "S_v must be a multiple of active_lanes");
    static_assert(S_v % gdn_block_dv == 0, "S_v must be a multiple of gdn_block_dv");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      warp_id  = threadIdx.y;

    // each warp owns gdn_d_v_per_warp rows of the state matrix; the warp's lanes shard the QK axis
    const uint32_t dv_base  = blockIdx.z * gdn_block_dv + warp_id * gdn_d_v_per_warp;
    const uint32_t dqk_base = lane * d_qk_per_lane;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst;

    // curr_state and state are not __restrict__: with cache fusion the state can be updated in place

    // input state holds s0 only: [S_v, S_v, H, n_seqs] - seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) - same per-(seq,head) offset.
    const int64_t state_off = ((int64_t) sequence * H + h_idx) * S_v * S_v;
    state      += state_off;
    curr_state += state_off;
    attn_data  += ((int64_t) sequence * n_tokens * H + h_idx) * S_v;

    auto load_qk_lane = [&] __device__(float(&reg)[d_qk_per_lane], const float * base) {
        if constexpr (S_v < warp_size) {
            reg[0] = (lane < active_lanes) ? base[lane] : 0.0f;
        } else {
            ggml_cuda_memcpy_1<d_qk_per_lane * sizeof(float)>(reg, base + dqk_base);
        }
    };
    auto store_qk_lane = [&] __device__(const float(&reg)[d_qk_per_lane], float * base) {
        if constexpr (S_v < warp_size) {
            if (lane < active_lanes) {
                base[lane] = reg[0];
            }
        } else {
            ggml_cuda_memcpy_1<d_qk_per_lane * sizeof(float)>(base + dqk_base, reg);
        }
    };

    ggml_cuda_pdl_sync();

    // state is stored transposed: M[r][c] = S[c][r]
    __align__(16) float s_tile[gdn_d_v_per_warp][d_qk_per_lane];
#pragma unroll
    for (int r = 0; r < gdn_d_v_per_warp; ++r) {
        load_qk_lane(s_tile[r], curr_state + (int64_t) (dv_base + r) * S_v);
    }

    __align__(16) float k_reg[d_qk_per_lane];
    load_qk_lane(k_reg, k + (int64_t) iq3 * sq3 + (int64_t) iq1 * sq1);

    for (int t = 0; t < n_tokens; ++t) {
        const float * v_t = v + (int64_t) sequence * sv3 + (int64_t) t * sv2 + (int64_t) h_idx * sv1;

        const int64_t gb_off   = (int64_t) sequence * sb3 + (int64_t) t * sb2 + (int64_t) h_idx * sb1;
        const float   beta_val = beta[gb_off];

        __align__(16) float alpha_lane[d_qk_per_lane];
        float               alpha_scalar = 0.0f;
        if constexpr (KDA) {
            __align__(16) float g_reg[d_qk_per_lane];
            load_qk_lane(g_reg, g + gb_off * S_v);
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                alpha_lane[c] = expf(g_reg[c]);
            }
        } else {
            alpha_scalar = expf(g[gb_off]);
        }

        // only the first gdn_d_v_per_warp lanes hold a real v[dv_base + lane]; broadcast via __shfl_sync
        float v_local = 0.0f;
        if (lane < gdn_d_v_per_warp) {
            v_local = v_t[dv_base + lane];
        }

        // stage A: state update
#pragma unroll
        for (int r = 0; r < gdn_d_v_per_warp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                if constexpr (KDA) {
                    partial += alpha_lane[c] * s_tile[r][c] * k_reg[c];
                } else {
                    partial += s_tile[r][c] * k_reg[c];
                }
            }
            partial = warp_reduce_sum<warp_size>(partial);

            const float v_r   = __shfl_sync(0xffffffff, v_local, r, warp_size);
            const float delta = beta_val * (v_r - (KDA ? 1.0f : alpha_scalar) * partial);

#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                if constexpr (KDA) {
                    s_tile[r][c] = alpha_lane[c] * s_tile[r][c] + delta * k_reg[c];
                } else {
                    s_tile[r][c] = alpha_scalar * s_tile[r][c] + delta * k_reg[c];
                }
            }
        }

        // prefetch k for next token while issuing the q load for this token
        if (t + 1 < n_tokens) {
            load_qk_lane(k_reg, k + (int64_t) iq3 * sq3 + (int64_t) (t + 1) * sq2 + (int64_t) iq1 * sq1);
        }

        __align__(16) float q_reg[d_qk_per_lane];
        load_qk_lane(q_reg, q + (int64_t) iq3 * sq3 + (int64_t) t * sq2 + (int64_t) iq1 * sq1);

        // stage B: attention output
        float attn_val = 0.0f;
#pragma unroll
        for (int r = 0; r < gdn_d_v_per_warp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                partial += s_tile[r][c] * q_reg[c];
            }
            partial = warp_reduce_sum<warp_size>(partial);
            if (lane == r) {
                attn_val = partial;
            }
        }

        if (lane < gdn_d_v_per_warp) {
            attn_data[dv_base + lane] = attn_val * scale;
        }
        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * slot_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < gdn_d_v_per_warp; ++r) {
                    store_qk_lane(s_tile[r], slot_state + (int64_t) (dv_base + r) * S_v);
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < gdn_d_v_per_warp; ++r) {
            store_qk_lane(s_tile[r], state + (int64_t) (dv_base + r) * S_v);
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    const int warp_size  = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int n_block_dv = (int) ((S_v + gdn_block_dv - 1) / gdn_block_dv);
    dim3      grid_dims(H, n_seqs, n_block_dv);
    dim3      block_dims(warp_size, gdn_num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// Shape-only half of the chunked predicate, split out because the buffer-type get_alloc_size hook
// has to size the chunked scratch. That sizing must NOT depend on which device is current: deciding
// from shape alone can only over-allocate on a device that ends up ineligible, never under-allocate
// (which would corrupt memory). See ggml_cuda_gdn_get_alloc_size.
bool ggml_cuda_gdn_chunked_shape_eligible(const ggml_tensor * dst) {
    if (dst->op != GGML_OP_GATED_DELTA_NET) {
        return false;
    }

    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_k     = dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];

    const int64_t S_v      = src_v->ne[0];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t neq0     = src_q->ne[0];
    const int64_t neq1     = src_q->ne[1];   // q head count
    const int64_t nev1     = src_v->ne[1];   // v head count
    const bool    kda      = (src_g->ne[0] == S_v);
    const int     K        = ggml_get_op_params_i32(dst, 0);

    // - not KDA; K > 1 runs chunked on the first n_tokens - (K - 1) tokens and recurrent on the rest
    // - Q/K/G/beta/state must be contiguous; V must be contiguous within each token (nb[0]/nb[1]
    //   packed) and packed across sequences (nb[3] == n_tokens*nb[2]), with an arbitrary per-token
    //   stride nb[2] (fused QKV view). The nb[3] check matches the chunked-entry assert: without it a
    //   view with inter-sequence padding would pass dispatch and then read the wrong batch slice.
    // - 128-wide heads, GQA-aligned head counts, n_tokens - (K - 1) >= 128
    return !kda && K >= 1
        && neq0 == 128 && S_v == 128 && nev1 % neq1 == 0
        && src_k->ne[1] == neq1
        && n_tokens - (K - 1) >= 128
        && ggml_is_contiguous(src_q) && ggml_is_contiguous(src_k) && ggml_is_contiguous(src_g)
        && src_v->nb[0] == ggml_type_size(src_v->type) && src_v->nb[1] == (size_t)S_v * ggml_type_size(src_v->type)
        && src_v->nb[3] == (size_t) n_tokens * src_v->nb[2]
        && ggml_is_contiguous(src_beta) && ggml_is_contiguous(src_state);
}

bool ggml_cuda_should_use_chunked_gdn(const ggml_tensor * dst) {
#ifdef GGML_CUDA_NO_GDN_CHUNK
    // Recurrent-only baseline build (see tools/bench_ab_*): everything routes to the recurrent
    // kernel, which also re-enables CUDA graphs for the op.
    GGML_UNUSED(dst);
    return false;
#else
    if (!ggml_cuda_gdn_chunked_shape_eligible(dst)) {
        return false;
    }
    // NVIDIA Ampere+ only (fp16 WMMA). The HIP/MUSA ggml_cuda_mma backend is intentionally not
    // dispatched until validated.
    const int cc_dev = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc_dev) && cc_dev >= GGML_CUDA_CC_AMPERE;
#endif // GGML_CUDA_NO_GDN_CHUNK
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // the recurrent kernel loads q/k rows with vectorized copies
    GGML_ASSERT(nbq1 % 16 == 0);
    GGML_ASSERT(nbq2 % 16 == 0);
    GGML_ASSERT(nbq3 % 16 == 0);

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // Route to the chunked prefill kernel when eligible.
    // Passes cache so the kernel can write the final state directly to the fused destination
    // (cache->data). Scratch lives in dst's own allocation, so its address is stable across CUDA
    // graph capture and replay.
    const bool use_chunked = ggml_cuda_should_use_chunked_gdn(dst);
    if (use_chunked && !keep_rs) {
        ggml_cuda_op_gated_delta_net_chunked(ctx, dst, cache);
        return;
    }

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (use_chunked) {
        // slot s holds the state s tokens before the end, so the chunked prefix writes slot K-1
        // and the recurrent kernel continues from it, filling slots K-2..0
        const int64_t n_pre = n_tokens - (K - 1);
        const int64_t D     = S_v * S_v * H;

        for (int64_t i3 = 0; i3 < n_seqs; ++i3) {
            const int64_t iq3 = i3 / rq3;

            const float * q_s = q_d + iq3*sq3;
            const float * k_s = k_d + iq3*sq3;
            const float * v_s = v_d + i3*sv3;
            const float * g_s = g_d + i3*sb3;
            const float * b_s = b_d + i3*sb3;
            float * out_s     = dst_d + i3*n_tokens*H*S_v;
            float * slot_last = state_d + (K - 1)*state_slot_stride + i3*D;

            ggml_cuda_op_gated_delta_net_chunked_impl(ctx, dst, out_s, slot_last,
                1, n_pre, H, neqk1, S_v, S_v, (n_pre + 15)/16,
                q_s, k_s, v_s, g_s, b_s, s_d + i3*D, scale, sv2, stream);

            launch_gated_delta_net<false, true>(
                q_s + n_pre*sq2, k_s + n_pre*sq2, v_s + n_pre*sv2, g_s + n_pre*sb2, b_s + n_pre*sb2,
                slot_last, out_s + n_pre*H*S_v, state_d + i3*D,
                S_v, H, K - 1, 1, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, 1, scale, state_slot_stride, K - 1, stream);
        }
        return;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
