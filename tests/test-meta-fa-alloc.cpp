// Meta backend (-sm tensor): the flash attention allocation is sized per device when K/V are split by heads
// (upstream #30317, see [TAG_META_FA_SCRATCH] in ggml-backend-meta.cpp).
//
// The simple devices wrap the CPU device with a buffer type that reserves the same f16 K/V copies as the CUDA one
// (ggml_cuda_flash_attn_ext_get_alloc_size with an MMA kernel), so the per-device size can be checked without a GPU.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

static constexpr int64_t D      = 128;
static constexpr int64_t H_KV   = 4;
static constexpr int64_t GQA    = 2;
static constexpr int64_t N_KV   = 256;
static constexpr int64_t N_TOK  = 8;
static const int64_t     HEADS_DEV0 = 1; // uneven: device 0 gets 1 KV head, device 1 gets 3

static size_t fake_cuda_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * t) {
    GGML_UNUSED(buft);
    size_t size = ggml_nbytes(t);
    if (t->op == GGML_OP_FLASH_ATTN_EXT) {
        uintptr_t end = (uintptr_t) t->data + ggml_nbytes(t);
        for (int i = 1; i <= 2; i++) {
            if (t->src[i]->type != GGML_TYPE_F16) {
                end  = GGML_PAD(end, 128);
                end += ggml_nelements(t->src[i])*ggml_type_size(GGML_TYPE_F16);
            }
        }
        size = end - (uintptr_t) t->data;
    }
    return size;
}

static ggml_backend_buffer_type wrapped_buft[2];
static ggml_backend_device       wrapped_dev[2];

static ggml_backend_buffer_type_t wrapped_get_buffer_type(ggml_backend_dev_t dev) {
    return &wrapped_buft[dev == &wrapped_dev[0] ? 0 : 1];
}

static ggml_backend_meta_split_state split_state(const ggml_tensor * t, void * ud) {
    GGML_UNUSED(ud);
    ggml_backend_meta_split_state ss = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    if (strcmp(t->name, "k_cache") == 0 || strcmp(t->name, "v_cache") == 0) {
        ss.axis  = GGML_BACKEND_SPLIT_AXIS_0;
        ss.ne[0] = D*HEADS_DEV0;
        ss.ne[1] = D*(H_KV - HEADS_DEV0);
    } else if (strcmp(t->name, "q") == 0) {
        ss.axis  = GGML_BACKEND_SPLIT_AXIS_2;
        ss.ne[0] = GQA*HEADS_DEV0;
        ss.ne[1] = GQA*(H_KV - HEADS_DEV0);
    }
    return ss;
}

static size_t expected_fa_size(const ggml_tensor * out, int64_t heads) {
    uintptr_t end = ggml_nbytes(out);
    for (int i = 0; i < 2; i++) {
        end  = GGML_PAD(end, 128);
        end += D*N_KV*heads*ggml_type_size(GGML_TYPE_F16);
    }
    return end;
}

struct fa_graph {
    ggml_context_ptr ctx;
    ggml_tensor * out = nullptr;
    ggml_cgraph * gf  = nullptr;
};

static fa_graph build_graph(ggml_tensor * q, ggml_tensor * k_cache, ggml_tensor * v_cache) {
    ggml_init_params params = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    fa_graph g;
    g.ctx.reset(ggml_init(params));
    // same layout as llama_kv_cache::get_k/get_v and the permute in llm_graph_context::build_attn_mha
    ggml_tensor * k = ggml_view_4d(g.ctx.get(), k_cache, D, H_KV, N_KV, 1,
            ggml_row_size(k_cache->type, D), k_cache->nb[1], k_cache->nb[1]*N_KV, 0);
    ggml_tensor * v = ggml_view_4d(g.ctx.get(), v_cache, D, H_KV, N_KV, 1,
            ggml_row_size(v_cache->type, D), v_cache->nb[1], v_cache->nb[1]*N_KV, 0);
    k = ggml_permute(g.ctx.get(), k, 0, 2, 1, 3);
    v = ggml_permute(g.ctx.get(), v, 0, 2, 1, 3);
    g.out = ggml_flash_attn_ext(g.ctx.get(), q, k, v, nullptr, 1.0f/sqrtf(D), 0.0f, 0.0f);
    g.gf = ggml_new_graph(g.ctx.get());
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}

int main() {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu);
    for (int j = 0; j < 2; j++) {
        wrapped_buft[j] = *ggml_backend_dev_buffer_type(cpu);
        wrapped_buft[j].iface.get_alloc_size = fake_cuda_alloc_size;
        wrapped_buft[j].device = &wrapped_dev[j];
        wrapped_dev[j] = *cpu;
        wrapped_dev[j].iface.get_buffer_type = wrapped_get_buffer_type;
    }
    ggml_backend_dev_t devs[2] = { &wrapped_dev[0], &wrapped_dev[1] };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs, 2, split_state, nullptr);
    ggml_backend_ptr meta(ggml_backend_dev_init(meta_dev, nullptr));
    GGML_ASSERT(meta);
    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta_dev);

    // static tensors: KV cache split by heads, Q split by heads
    ggml_init_params sparams = { 8*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr sctx(ggml_init(sparams));
    ggml_tensor * k_cache = ggml_new_tensor_2d(sctx.get(), GGML_TYPE_Q8_0, D*H_KV, N_KV);
    ggml_tensor * v_cache = ggml_new_tensor_2d(sctx.get(), GGML_TYPE_Q4_0, D*H_KV, N_KV);
    ggml_tensor * q       = ggml_new_tensor_4d(sctx.get(), GGML_TYPE_F32, D, N_TOK, H_KV*GQA, 1);
    ggml_set_name(k_cache, "k_cache");
    ggml_set_name(v_cache, "v_cache");
    ggml_set_name(q, "q");
    ggml_backend_buffer_ptr sbuf(ggml_backend_alloc_ctx_tensors_from_buft(sctx.get(), meta_buft));
    GGML_ASSERT(sbuf);

    std::vector<float> q_data(ggml_nelements(q));
    std::vector<float> k_data(ggml_nelements(k_cache));
    std::vector<float> v_data(ggml_nelements(v_cache));
    for (size_t i = 0; i < q_data.size(); i++) { q_data[i] = sinf(0.37f*i); }
    for (size_t i = 0; i < k_data.size(); i++) { k_data[i] = cosf(0.11f*i); v_data[i] = sinf(0.23f*i + 1.0f); }
    std::vector<uint8_t> k_q(ggml_nbytes(k_cache));
    std::vector<uint8_t> v_q(ggml_nbytes(v_cache));
    ggml_quantize_chunk(k_cache->type, k_data.data(), k_q.data(), 0, N_KV, D*H_KV, nullptr);
    ggml_quantize_chunk(v_cache->type, v_data.data(), v_q.data(), 0, N_KV, D*H_KV, nullptr);
    ggml_backend_tensor_set(k_cache, k_q.data(), 0, k_q.size());
    ggml_backend_tensor_set(v_cache, v_q.data(), 0, v_q.size());
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size()*sizeof(float));

    fa_graph g = build_graph(q, k_cache, v_cache);

    // before allocation: the K/V views have no buffer yet, the split is probed from the cache
    const size_t full = expected_fa_size(g.out, H_KV);
    const size_t want = expected_fa_size(g.out, H_KV - HEADS_DEV0);
    const size_t got  = ggml_backend_buft_get_alloc_size(meta_buft, g.out);
    printf("FA alloc size: full %zu, per device %zu, meta %zu\n", full, want, got);
    const char * env_full = getenv("GGML_META_FA_SCRATCH_FULL");
    if (env_full != nullptr && atoi(env_full) != 0) {
        GGML_ASSERT(got == full);
    } else {
        GGML_ASSERT(got == want && want < full);
    }

    ggml_gallocr_ptr galloc(ggml_gallocr_new(meta_buft));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc.get(), g.gf));
    // after allocation the views are initialized: same size from the cached split states
    GGML_ASSERT(ggml_backend_buft_get_alloc_size(meta_buft, g.out) == got);
    GGML_ASSERT(ggml_backend_graph_compute(meta.get(), g.gf) == GGML_STATUS_SUCCESS);
    std::vector<float> out_meta(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, out_meta.data(), 0, out_meta.size()*sizeof(float));

    // reference on one CPU device
    ggml_backend_ptr cpu_backend(ggml_backend_dev_init(cpu, nullptr));
    ggml_context_ptr rctx(ggml_init(sparams));
    ggml_tensor * rk = ggml_dup_tensor(rctx.get(), k_cache);
    ggml_tensor * rv = ggml_dup_tensor(rctx.get(), v_cache);
    ggml_tensor * rq = ggml_dup_tensor(rctx.get(), q);
    ggml_backend_buffer_ptr rbuf(ggml_backend_alloc_ctx_tensors(rctx.get(), cpu_backend.get()));
    ggml_backend_tensor_set(rk, k_q.data(), 0, k_q.size());
    ggml_backend_tensor_set(rv, v_q.data(), 0, v_q.size());
    ggml_backend_tensor_set(rq, q_data.data(), 0, q_data.size()*sizeof(float));
    fa_graph r = build_graph(rq, rk, rv);
    ggml_gallocr_ptr ralloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend.get())));
    GGML_ASSERT(ggml_gallocr_alloc_graph(ralloc.get(), r.gf));
    GGML_ASSERT(ggml_backend_graph_compute(cpu_backend.get(), r.gf) == GGML_STATUS_SUCCESS);
    std::vector<float> out_ref(ggml_nelements(r.out));
    ggml_backend_tensor_get(r.out, out_ref.data(), 0, out_ref.size()*sizeof(float));

    double max_err = 0.0;
    for (size_t i = 0; i < out_ref.size(); i++) {
        max_err = std::max(max_err, (double) fabsf(out_ref[i] - out_meta[i]));
    }
    printf("max abs diff meta vs single device: %g\n", max_err);
    GGML_ASSERT(max_err < 1e-4);

    // mirrored K/V keep the full size
    ggml_context_ptr mctx(ggml_init(sparams));
    ggml_tensor * k2 = ggml_new_tensor_2d(mctx.get(), GGML_TYPE_Q8_0, D*H_KV, N_KV);
    ggml_tensor * v2 = ggml_new_tensor_2d(mctx.get(), GGML_TYPE_Q4_0, D*H_KV, N_KV);
    ggml_tensor * q2 = ggml_new_tensor_4d(mctx.get(), GGML_TYPE_F32, D, N_TOK, H_KV*GQA, 1);
    ggml_set_name(k2, "k_mirrored");
    ggml_set_name(v2, "v_mirrored");
    ggml_set_name(q2, "q_mirrored");
    ggml_backend_buffer_ptr mbuf(ggml_backend_alloc_ctx_tensors_from_buft(mctx.get(), meta_buft));
    fa_graph m = build_graph(q2, k2, v2);
    GGML_ASSERT(ggml_backend_buft_get_alloc_size(meta_buft, m.out) == expected_fa_size(m.out, H_KV));

    printf("OK\n");
    return 0;
}
