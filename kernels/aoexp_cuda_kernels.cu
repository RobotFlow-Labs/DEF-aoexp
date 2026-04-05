/*
 * AO-Exp Custom CUDA Kernels
 * ===========================
 * 1. svd_nuclear_prox: Fused nuclear norm proximal operator with Lambert W
 * 2. fused_mask_ce: Fused foreground/background mask cross-entropy loss
 *
 * Target: sm_89 (NVIDIA L4), CUDA 12.8, PyTorch cu128
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================
// Kernel 1: Lambert W0 approximation + proximal operator
// Applied per singular value after SVD decomposition
// ============================================================

__device__ __forceinline__ float lambert_w0_device(float z) {
    // Halley's method with log initialization
    if (z < 0.0f) return 0.0f;
    if (z < 2.0f) {
        float w = z / (1.0f + z);
        // 3 Halley iterations
        for (int i = 0; i < 3; i++) {
            float ew = expf(w);
            float wew = w * ew;
            float fp = ew * (w + 1.0f);
            float fpp = ew * (w + 2.0f);
            float f = wew - z;
            w -= f / (fp - f * fpp / (2.0f * fp));
        }
        return w;
    }
    float lz = logf(z);
    float llz = logf(lz);
    float w = lz - llz + llz / lz;
    for (int i = 0; i < 3; i++) {
        float ew = expf(w);
        float wew = w * ew;
        float fp = ew * (w + 1.0f);
        float fpp = ew * (w + 2.0f);
        float f = wew - z;
        w -= f / (fp - f * fpp / (2.0f * fp));
    }
    return w;
}

__global__ void svd_nuclear_prox_kernel(
    const float* __restrict__ s_in,    // Input singular values [N]
    float* __restrict__ s_out,         // Output singular values [N]
    const float lambda1,               // Nuclear norm weight
    const float lambda2,               // Frobenius weight
    const float t,                     // Current iteration
    const float alpha,                 // Adaptive step size
    const float beta_const,            // Beta constant (1.0)
    const int N,                       // Number of singular values
    const int k                        // Top-k truncation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float s_y = s_in[idx];

    float s_new;
    if (lambda2 == 0.0f) {
        float log_term = logf(s_y / beta_const + 1.0f);
        float thresh = lambda1 * t / alpha;
        float val = log_term - thresh;
        s_new = (val > 0.0f) ? (beta_const * expf(val) - beta_const) : 0.0f;
    } else {
        float a = beta_const;
        float b = lambda2 * t / alpha;
        float c_val = lambda1 * t / alpha - logf(s_y / beta_const + 1.0f);
        c_val = fminf(c_val, 0.0f);  // clamp max=0
        float abc = logf(a * b) + a * b - c_val;

        if (abc >= 15.0f) {
            float log_abc = logf(abc);
            float log_log_abc = logf(log_abc);
            s_new = (log_abc - log_log_abc + log_log_abc / log_abc) / b - a;
        } else {
            s_new = lambert_w0_device(expf(abc)) / b - a;
        }
    }

    // Top-k truncation: zero out values beyond k
    s_out[idx] = (idx < k) ? fmaxf(s_new, 0.0f) : 0.0f;
}

torch::Tensor svd_nuclear_prox(
    torch::Tensor s_in,
    float lambda1,
    float lambda2,
    float t,
    float alpha,
    float beta_const,
    int k
) {
    TORCH_CHECK(s_in.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(s_in.dim() == 1, "Input must be 1D");

    int N = s_in.size(0);
    auto s_out = torch::zeros_like(s_in);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    svd_nuclear_prox_kernel<<<blocks, threads>>>(
        s_in.data_ptr<float>(),
        s_out.data_ptr<float>(),
        lambda1, lambda2, t, alpha, beta_const,
        N, k
    );

    return s_out;
}


// ============================================================
// Kernel 2: Fused foreground/background mask cross-entropy
// Computes both losses in a single kernel pass
// ============================================================

__global__ void fused_mask_ce_kernel(
    const float* __restrict__ adv_mask,    // Adversarial mask [H*W]
    const float* __restrict__ clean_mask,  // Clean mask [H*W]
    float* __restrict__ fg_loss,           // Output: foreground CE loss [1]
    float* __restrict__ bg_loss,           // Output: background CE loss [1]
    int* __restrict__ fg_count,            // Output: foreground pixel count [1]
    int* __restrict__ bg_count,            // Output: background pixel count [1]
    const int N                            // Total pixels H*W
) {
    // Shared memory for block-level reduction
    extern __shared__ float sdata[];
    float* s_fg = sdata;
    float* s_bg = sdata + blockDim.x;
    int* s_fg_cnt = (int*)(sdata + 2 * blockDim.x);
    int* s_bg_cnt = (int*)(sdata + 2 * blockDim.x) + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_fg = 0.0f;
    float local_bg = 0.0f;
    int local_fg_cnt = 0;
    int local_bg_cnt = 0;

    if (idx < N) {
        float p = adv_mask[idx];
        p = fmaxf(fminf(p, 1.0f - 1e-7f), 1e-7f);
        float target = (clean_mask[idx] > 0.0f) ? 1.0f : 0.0f;

        // Cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
        float ce = -(target * logf(p) + (1.0f - target) * logf(1.0f - p));

        if (target > 0.5f) {
            local_fg = ce;
            local_fg_cnt = 1;
        } else {
            local_bg = ce;
            local_bg_cnt = 1;
        }
    }

    s_fg[tid] = local_fg;
    s_bg[tid] = local_bg;
    s_fg_cnt[tid] = local_fg_cnt;
    s_bg_cnt[tid] = local_bg_cnt;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_fg[tid] += s_fg[tid + s];
            s_bg[tid] += s_bg[tid + s];
            s_fg_cnt[tid] += s_fg_cnt[tid + s];
            s_bg_cnt[tid] += s_bg_cnt[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(fg_loss, s_fg[0]);
        atomicAdd(bg_loss, s_bg[0]);
        atomicAdd(fg_count, s_fg_cnt[0]);
        atomicAdd(bg_count, s_bg_cnt[0]);
    }
}

std::vector<torch::Tensor> fused_mask_ce(
    torch::Tensor adv_mask,
    torch::Tensor clean_mask
) {
    TORCH_CHECK(adv_mask.is_cuda(), "adv_mask must be CUDA");
    TORCH_CHECK(clean_mask.is_cuda(), "clean_mask must be CUDA");

    int N = adv_mask.numel();
    auto options_f = torch::TensorOptions().dtype(torch::kFloat32).device(adv_mask.device());
    auto options_i = torch::TensorOptions().dtype(torch::kInt32).device(adv_mask.device());

    auto fg_loss = torch::zeros(1, options_f);
    auto bg_loss = torch::zeros(1, options_f);
    auto fg_count = torch::zeros(1, options_i);
    auto bg_count = torch::zeros(1, options_i);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int smem = threads * (2 * sizeof(float) + 2 * sizeof(int));

    fused_mask_ce_kernel<<<blocks, threads, smem>>>(
        adv_mask.contiguous().data_ptr<float>(),
        clean_mask.contiguous().data_ptr<float>(),
        fg_loss.data_ptr<float>(),
        bg_loss.data_ptr<float>(),
        fg_count.data_ptr<int>(),
        bg_count.data_ptr<int>(),
        N
    );

    // Normalize by counts
    float fg_n = std::max(fg_count.item<int>(), 1);
    float bg_n = std::max(bg_count.item<int>(), 1);

    return {fg_loss / fg_n, bg_loss / bg_n};
}


// ============================================================
// PyBind11 module
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("svd_nuclear_prox", &svd_nuclear_prox,
          "SVD nuclear norm proximal operator with Lambert W (CUDA)");
    m.def("fused_mask_ce", &fused_mask_ce,
          "Fused foreground/background mask cross-entropy loss (CUDA)");
}
