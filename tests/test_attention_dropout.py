import copy
import unittest

import torch
from torch import nn

from k_diffusion import layers
from k_diffusion.models.image_v1 import ImageDenoiserModelV1


class TestAttentionDropout(unittest.TestCase):
    @staticmethod
    def make_attention(kind, dropout, dtype=torch.float32, heads=2):
        torch.manual_seed(310)

        def norm(channels):
            return layers.AdaGN(4, channels, 1)

        if kind == "self":
            module = layers.SelfAttention2d(8, heads, norm, dropout)
        else:
            module = layers.CrossAttention2d(8, 6, heads, norm, dropout)
        module = module.to(dtype=dtype)
        with torch.no_grad():
            nn.init.normal_(module.out_proj.weight, std=0.2)
            module.out_proj.bias.zero_()
        image = torch.randn(2, 8, 3, 4, dtype=dtype)
        cond = {
            "cond": torch.randn(2, 4, dtype=dtype),
            "cross": torch.randn(2, 5, 6, dtype=dtype),
            "cross_padding": torch.tensor(
                [[0, 0, 0, 0, 1], [0, 1, 0, 0, 0]], dtype=dtype
            ),
        }
        return module, image, cond

    def test_eval_matches_dropout_free_values_and_gradients(self):
        for kind in ("self", "cross"):
            for dtype in (torch.float32, torch.float64):
                for dropout in (0.2, 1.0):
                    with self.subTest(kind=kind, dtype=dtype, dropout=dropout):
                        module, image, cond = self.make_attention(kind, dropout, dtype)
                        module.eval()
                        reference = copy.deepcopy(module)
                        reference.dropout.p = 0.0
                        actual_input = image.clone().requires_grad_()
                        reference_input = image.clone().requires_grad_()
                        actual = module(actual_input, cond)
                        expected = reference(reference_input, cond)
                        torch.testing.assert_close(
                            actual, expected, rtol=1e-5, atol=1e-6
                        )
                        weights = torch.linspace(
                            -0.7, 0.9, actual.numel(), dtype=dtype
                        ).reshape_as(actual)
                        actual_gradients = torch.autograd.grad(
                            (actual * weights).sum(),
                            (actual_input, *module.parameters()),
                        )
                        reference_gradients = torch.autograd.grad(
                            (expected * weights).sum(),
                            (reference_input, *reference.parameters()),
                        )
                        for gradient, expected_gradient in zip(
                            actual_gradients, reference_gradients
                        ):
                            torch.testing.assert_close(
                                gradient, expected_gradient, rtol=1e-5, atol=1e-6
                            )

    def test_eval_is_repeatable_and_does_not_consume_rng(self):
        for kind in ("self", "cross"):
            with self.subTest(kind=kind):
                module, image, cond = self.make_attention(kind, 0.2)
                module.eval()
                rng = torch.random.get_rng_state()
                first = module(image, cond)
                torch.testing.assert_close(
                    torch.random.get_rng_state(), rng, rtol=0, atol=0
                )
                second = module(image, cond)
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_training_keeps_dropout_active(self):
        for kind in ("self", "cross"):
            with self.subTest(kind=kind):
                module, image, cond = self.make_attention(kind, 1.0)
                module.train()
                # Dropping every attention weight leaves only the residual.
                torch.testing.assert_close(module(image, cond), image, rtol=0, atol=0)
                module.eval()
                self.assertFalse(torch.equal(module(image, cond), image))

    def test_dropout_child_mode_is_respected(self):
        for kind in ("self", "cross"):
            with self.subTest(kind=kind):
                module, image, cond = self.make_attention(kind, 1.0)
                reference = copy.deepcopy(module)
                reference.dropout.p = 0.0
                module.train()
                module.dropout.eval()
                torch.testing.assert_close(
                    module(image, cond), reference(image, cond), rtol=0, atol=0
                )
                module.eval()
                module.dropout.train()
                torch.testing.assert_close(module(image, cond), image, rtol=0, atol=0)

    def test_zero_dropout_controls_in_both_modes(self):
        for kind in ("self", "cross"):
            for training in (False, True):
                for heads in (1, 2):
                    with self.subTest(kind=kind, training=training, heads=heads):
                        module, image, cond = self.make_attention(
                            kind, 0.0, heads=heads
                        )
                        module.train(training)
                        rng = torch.random.get_rng_state()
                        actual = module(image, cond)
                        expected = module(image, cond)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(
                            torch.random.get_rng_state(), rng, rtol=0, atol=0
                        )

    def test_nonzero_training_dropout_is_stochastic_and_seeded(self):
        for kind in ("self", "cross"):
            with self.subTest(kind=kind):
                module, image, cond = self.make_attention(kind, 0.2)
                module.train()
                torch.manual_seed(410)
                first = module(image, cond)
                torch.manual_seed(410)
                repeat = module(image, cond)
                torch.testing.assert_close(first, repeat, rtol=0, atol=0)
                torch.manual_seed(411)
                second = module(image, cond)
                self.assertFalse(torch.equal(first, second))

    def test_image_v1_denoising_matches_dropout_free_model(self):
        for cross_conditioned in (False, True):
            with self.subTest(cross_conditioned=cross_conditioned):
                torch.manual_seed(510)
                model = ImageDenoiserModelV1(
                    c_in=3,
                    feats_in=8,
                    depths=[1, 1],
                    channels=[8, 16],
                    self_attn_depths=[True, True],
                    cross_attn_depths=[True, True],
                    cross_cond_dim=6 if cross_conditioned else 0,
                    dropout_rate=0.15,
                )
                with torch.no_grad():
                    for module in model.modules():
                        if isinstance(
                            module, (layers.SelfAttention2d, layers.CrossAttention2d)
                        ):
                            nn.init.normal_(module.out_proj.weight, std=0.1)
                    nn.init.normal_(model.proj_out.weight, std=0.1)
                model.eval()
                reference = copy.deepcopy(model)
                for module in reference.modules():
                    if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                        module.p = 0.0
                image = torch.randn(2, 3, 8, 8)
                sigma = torch.tensor([0.5, 2.0])
                kwargs = {}
                if cross_conditioned:
                    kwargs = {
                        "cross_cond": torch.randn(2, 5, 6),
                        "cross_cond_padding": torch.tensor(
                            [[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
                        ),
                    }
                actual = model(image, sigma, **kwargs)
                expected = reference(image, sigma, **kwargs)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(
                    actual, model(image, sigma, **kwargs), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
