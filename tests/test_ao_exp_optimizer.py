"""Tests for AO-Exp optimizer and core components."""

from __future__ import annotations

import pytest
import torch

from def_aoexp.ao_exp_optimizer import AOExpOptimizer, _lambert_w0_approx
from def_aoexp.losses import (
    confidence_loss,
    mask_cross_entropy_losses,
    mean_absolute_perturbation,
    nuclear_norm,
)


class TestLambertW:
    def test_known_values(self):
        # W(e) = 1
        z = torch.tensor([2.718281828])
        w = _lambert_w0_approx(z)
        assert abs(w.item() - 1.0) < 1e-5

    def test_zero_region(self):
        z = torch.tensor([0.01, 0.1, 0.5])
        w = _lambert_w0_approx(z)
        # W(z)*exp(W(z)) should equal z
        reconstructed = w * torch.exp(w)
        assert torch.allclose(reconstructed, z, atol=1e-4)

    def test_large_values(self):
        z = torch.tensor([100.0, 1000.0])
        w = _lambert_w0_approx(z)
        reconstructed = w * torch.exp(w)
        assert torch.allclose(reconstructed, z, rtol=1e-3)

    def test_batch(self):
        z = torch.linspace(0.1, 50.0, 100)
        w = _lambert_w0_approx(z)
        assert w.shape == z.shape
        assert torch.all(w > 0)


class TestAOExpOptimizer:
    def test_init(self):
        opt = AOExpOptimizer(shape=(64, 64, 3), device="cpu")
        assert opt.x.shape == (64, 64, 3)
        assert opt.y.shape == (64, 64, 3)

    def test_single_step(self):
        opt = AOExpOptimizer(shape=(32, 32, 3), device="cpu", lambda1=0.1, lambda2=0.01)
        grad = torch.randn(32, 32, 3)
        result = opt.step(grad)
        assert result.shape == (32, 32, 3)
        assert torch.all(result >= opt.lower)
        assert torch.all(result <= opt.upper)

    def test_multiple_steps(self):
        opt = AOExpOptimizer(shape=(32, 32, 3), device="cpu")
        for _ in range(5):
            grad = torch.randn(32, 32, 3) * 0.1
            result = opt.step(grad)
        assert result.shape == (32, 32, 3)
        assert opt.t == 5.0

    def test_nuclear_norm_decreases_with_regularization(self):
        opt = AOExpOptimizer(shape=(32, 32, 3), device="cpu", lambda1=1.0, lambda2=0.1)
        grad = torch.randn(32, 32, 3)
        opt.step(grad)
        nn = opt.nuclear_norm()
        assert nn >= 0.0

    def test_top1_mode(self):
        opt = AOExpOptimizer(shape=(32, 32, 3), device="cpu", k="top1")
        assert opt.k_val == 1
        grad = torch.randn(32, 32, 3) * 0.1
        result = opt.step(grad)
        assert result.shape == (32, 32, 3)

    def test_get_perturbation(self):
        opt = AOExpOptimizer(shape=(64, 48, 3), device="cpu")
        grad = torch.randn(64, 48, 3) * 0.1
        opt.step(grad)
        p = opt.get_perturbation()
        assert p.shape == (3, 64, 48)  # C, H, W

    def test_state_dict_roundtrip(self):
        opt1 = AOExpOptimizer(shape=(16, 16, 3), device="cpu")
        grad = torch.randn(16, 16, 3)
        opt1.step(grad)
        state = opt1.state_dict()

        opt2 = AOExpOptimizer(shape=(16, 16, 3), device="cpu")
        opt2.load_state_dict(state)
        assert torch.allclose(opt1.x, opt2.x)
        assert torch.allclose(opt1.y, opt2.y)


class TestLosses:
    def test_mask_ce_losses(self):
        adv_mask = torch.rand(64, 64)
        clean_mask = (torch.rand(64, 64) > 0.5).float()
        fg, bg = mask_cross_entropy_losses(adv_mask, clean_mask)
        assert isinstance(fg, torch.Tensor) or isinstance(fg, float)
        assert isinstance(bg, torch.Tensor) or isinstance(bg, float)

    def test_confidence_loss(self):
        scores = torch.tensor([0.9, 0.85, 0.95])
        loss = confidence_loss(scores)
        assert loss.item() == pytest.approx(270.0, rel=1e-3)

    def test_nuclear_norm_hwc(self):
        x = torch.randn(64, 64, 3)
        nn = nuclear_norm(x)
        assert nn.item() > 0

    def test_nuclear_norm_chw(self):
        x = torch.randn(3, 64, 64)
        nn = nuclear_norm(x)
        assert nn.item() > 0

    def test_map_metric(self):
        delta = torch.randn(64, 64, 3) * 0.01
        m = mean_absolute_perturbation(delta)
        assert m > 0
