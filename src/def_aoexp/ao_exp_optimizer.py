"""AO-Exp Optimizer: Adaptive Optimistic Exponentiated Gradient with nuclear norm.

GPU-accelerated PyTorch reimplementation of the AO-Exp algorithm from:
'Structured Universal Adversarial Attacks on Object Detection for Video Sequences'
(Jacob, Shao, Kasneci — GCPR 2025, arXiv:2510.14460)

Key improvements over reference numpy implementation:
- Full GPU tensor operations (no numpy SVD)
- torch.linalg.svd for batched SVD
- Lambert W approximation in pure PyTorch (no scipy dependency at runtime)
- AMP-compatible gradient computation
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _lambert_w0_approx(z: torch.Tensor) -> torch.Tensor:
    """Approximate principal branch Lambert W function W_0(z) for z >= 0.

    Uses Halley's method with log-based initialization for numerical stability.
    Accurate to ~1e-7 relative error for z > 0.
    """
    # Initial guess via log approximation
    log_z = torch.log(z.clamp(min=1e-30))
    log_log_z = torch.log(log_z.clamp(min=1e-30))
    w = log_z - log_log_z + log_log_z / log_z

    # For small z, use simpler initialization
    small = z < 2.0
    w = torch.where(small, z / (1.0 + z), w)

    # 4 iterations of Halley's method: very fast convergence
    for _ in range(4):
        ew = torch.exp(w)
        wew = w * ew
        f = wew - z
        fp = ew * (w + 1.0)
        fpp = ew * (w + 2.0)
        w = w - f / (fp - f * fpp / (2.0 * fp))

    return w


class AOExpOptimizer:
    """Adaptive Optimistic Exponentiated Gradient optimizer for UAP generation.

    Maintains per-channel SVD state and performs mirror descent updates
    with nuclear + Frobenius norm regularization.

    Args:
        shape: Perturbation shape (H, W, C) — typically (640, 640, 3).
        lambda1: Nuclear norm regularization weight.
        lambda2: Frobenius norm regularization weight.
        eta: Base learning rate for exponentiated gradient.
        k: Singular value truncation. Float in (0,1] = fraction of min(H,W).
           String 'top1' = keep only top singular value.
        lower: Lower clamp bound for perturbation values.
        upper: Upper clamp bound for perturbation values.
        device: Torch device.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        lambda1: float = 0.1,
        lambda2: float = 0.01,
        eta: float = 0.55,
        k: float | str = 1.0,
        lower: float = -1.0,
        upper: float = 1.0,
        device: str | torch.device = "cuda:0",
    ):
        self.device = torch.device(device)
        h, w, c = shape
        self.h, self.w, self.c = h, w, c
        self.d = min(h, w)

        # State tensors: x is the iterate, y is the output (weighted average)
        self.x = torch.zeros(h, w, c, device=self.device)
        self.y = torch.zeros(h, w, c, device=self.device)
        self.h_prev = torch.zeros(h, w, c, device=self.device)  # previous gradient

        # Per-channel accumulated singular value norms for adaptive step size
        self.lam_acc = torch.zeros(c, device=self.device)

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eta = eta
        self.lower = lower
        self.upper = upper

        # Singular value truncation
        if k == "top1":
            self.k_val = 1
        else:
            self.k_val = max(1, int(float(k) * self.d))

        self.t = 0.0
        self.beta = 0.0

    @torch.no_grad()
    def step(self, gradient: torch.Tensor) -> torch.Tensor:
        """Perform one AO-Exp update step.

        Args:
            gradient: Gradient of loss w.r.t. perturbation, shape (H, W, C).

        Returns:
            Updated perturbation y, shape (H, W, C).
        """
        self.t += 1.0
        self.beta += self.t

        g = gradient.to(self.device)

        for ch in range(self.c):
            g_ch = g[:, :, ch]
            h_ch = self.h_prev[:, :, ch]

            # Adaptive step size: accumulate top singular value of gradient difference
            diff = g_ch - h_ch
            s_top = torch.linalg.svdvals(diff)[0]
            self.lam_acc[ch] += (self.t * s_top) ** 2

            alpha = (
                torch.sqrt(self.lam_acc[ch])
                / (torch.sqrt(torch.log(torch.tensor(self.d + 1.0, device=self.device))))
                * self.eta
            )
            alpha = alpha.clamp(min=1e-6)

            beta_const = 1.0

            # SVD of current iterate
            u_x, s_x, vh_x = torch.linalg.svd(self.x[:, :, ch], full_matrices=False)

            # Optimistic gradient step in log-singular-value space
            log_term = alpha * torch.log(s_x / beta_const + 1.0)
            z_mat = (
                u_x * log_term.unsqueeze(0) @ vh_x
                - self.t * g_ch
                + self.t * h_ch
                - (self.t + 1.0) * g_ch
            )

            # SVD of z to get new singular values
            u_z, s_z, vh_z = torch.linalg.svd(z_mat, full_matrices=False)

            # Exponentiated map back
            s_y = beta_const * torch.exp(s_z / alpha) - beta_const

            # Proximal operator for nuclear + Frobenius regularization
            if self.lambda2 == 0.0:
                s_new = beta_const * torch.exp(
                    F.relu(torch.log(s_y / beta_const + 1.0) - self.lambda1 * self.t / alpha)
                ) - beta_const
            else:
                a = beta_const
                b = self.lambda2 * self.t / alpha
                c_val = torch.clamp(
                    self.lambda1 * self.t / alpha - torch.log(s_y / beta_const + 1.0),
                    max=0.0,
                )
                ab = torch.as_tensor(a * b, device=self.device)
                abc = torch.log(ab) + ab - c_val

                # Use Lambert W for moderate values, log approximation for large
                large_mask = abc >= 15.0
                w_large = torch.log(abc) - torch.log(torch.log(abc.clamp(min=1e-30))) + (
                    torch.log(torch.log(abc.clamp(min=1e-30))) / torch.log(abc.clamp(min=1e-30))
                )
                w_small = _lambert_w0_approx(torch.exp(abc))
                w_val = torch.where(large_mask, w_large, w_small)
                s_new = w_val / b - a

            # Top-k truncation
            s_trunc = torch.zeros_like(s_new)
            k = min(self.k_val, len(s_new))
            s_trunc[:k] = s_new[:k]

            # Reconstruct perturbation channel
            new_x = u_z * s_trunc.unsqueeze(0) @ vh_z
            self.x[:, :, ch] = new_x.clamp(self.lower, self.upper)

            # Weighted average for output
            self.y[:, :, ch] = (
                (self.t / self.beta) * self.x[:, :, ch]
                + ((self.beta - self.t) / self.beta) * self.y[:, :, ch]
            )

        # Store gradient for next optimistic step
        self.h_prev.copy_(g)

        return self.y.clone()

    def get_perturbation(self) -> torch.Tensor:
        """Return current perturbation as (C, H, W) tensor for image application."""
        return self.y.permute(2, 0, 1).clone()

    def nuclear_norm(self) -> float:
        """Compute mean nuclear norm across channels."""
        norms = []
        for ch in range(self.c):
            s = torch.linalg.svdvals(self.y[:, :, ch])
            norms.append(s.sum().item())
        return sum(norms) / len(norms)

    def state_dict(self) -> dict:
        return {
            "x": self.x.cpu(),
            "y": self.y.cpu(),
            "h_prev": self.h_prev.cpu(),
            "lam_acc": self.lam_acc.cpu(),
            "t": self.t,
            "beta": self.beta,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "eta": self.eta,
            "k_val": self.k_val,
        }

    def load_state_dict(self, state: dict) -> None:
        self.x = state["x"].to(self.device)
        self.y = state["y"].to(self.device)
        self.h_prev = state["h_prev"].to(self.device)
        self.lam_acc = state["lam_acc"].to(self.device)
        self.t = state["t"]
        self.beta = state["beta"]
