import math
import unittest
from decimal import Decimal, localcontext

import torch

from k_diffusion import sampling


class AncestralStepTest(unittest.TestCase):
    # Scales keep both inputs and correct outputs representable. The original
    # fourth-power intermediate overflows or underflows at the extreme scales.
    cases = (
        (torch.float16, (1.0, 1e-3)),
        (torch.float32, (1.0, 1e10, 1e-14, 1e25, 1e-25)),
        (torch.float64, (1.0, 1e100, 1e-100, 1e200, 1e-200)),
    )

    @staticmethod
    def reference(sigma_from, sigma_to, eta):
        # Evaluate the variance definition independently, with decimal
        # arithmetic rather than the tensor implementation's precision.
        with localcontext() as context:
            # Tiny b/a ratios need enough digits to retain b² - sigma_up².
            context.prec = 1200
            a = Decimal.from_float(float(sigma_from))
            b = Decimal.from_float(float(sigma_to))
            e = Decimal.from_float(float(eta))
            up = min(b, e * (b * b * (a * a - b * b) / (a * a)).sqrt())
            down = (b * b - up * up).sqrt()
            return float(down / b), float(up / b)

    @staticmethod
    def tolerance(dtype):
        return {torch.float16: 4e-3, torch.bfloat16: 2e-2,
                torch.float32: 3e-6, torch.float64: 2e-13}[dtype]

    def test_variance_definition_across_dtypes_and_scales(self):
        for dtype, scales in self.cases:
            for scale in scales:
                sigma_from = torch.tensor(80 * scale, dtype=dtype)
                sigma_to = torch.tensor(60 * scale, dtype=dtype)
                for eta in (0.25, 0.5, 1.0, 2.0):
                    with self.subTest(dtype=dtype, scale=scale, eta=eta):
                        down, up = sampling.get_ancestral_step(sigma_from, sigma_to, eta)
                        expected = self.reference(sigma_from, sigma_to, eta)
                        actual = torch.stack((down / sigma_to, up / sigma_to))
                        torch.testing.assert_close(
                            actual, actual.new_tensor(expected),
                            rtol=self.tolerance(dtype), atol=self.tolerance(dtype),
                        )
                        self.assertEqual(down.dtype, dtype)
                        self.assertEqual(up.dtype, dtype)
                        self.assertEqual(down.device, sigma_from.device)
                        self.assertEqual(up.device, sigma_from.device)

    def test_default_down_step_retains_small_representable_ratio(self):
        for dtype, sigma_from, sigma_to in (
            (torch.float32, 1e30, 1e5),
            (torch.float32, 80.0, 0.008),
            (torch.float64, 1e300, 1e100),
        ):
            with self.subTest(dtype=dtype, sigma_from=sigma_from, sigma_to=sigma_to):
                a = torch.tensor(sigma_from, dtype=dtype)
                b = torch.tensor(sigma_to, dtype=dtype)
                down, up = sampling.get_ancestral_step(a, b)
                expected = b * (b / a)
                self.assertGreater(expected.item(), 0)
                torch.testing.assert_close(down, expected, rtol=self.tolerance(dtype), atol=0)
                self.assertTrue(torch.isfinite(up))

    def test_python_scalars_follow_the_same_variance_definition(self):
        for scale in (1.0, 1e100, 1e-100, 1e200, 1e-200):
            for eta in (0.25, 0.5, 1.0, 2.0):
                with self.subTest(scale=scale, eta=eta):
                    a, b = 80 * scale, 60 * scale
                    down, up = sampling.get_ancestral_step(a, b, eta)
                    expected = self.reference(a, b, eta)
                    self.assertAlmostEqual(down / b, expected[0], places=13)
                    self.assertAlmostEqual(up / b, expected[1], places=13)

    def test_zero_eta_preserves_the_deterministic_contract(self):
        for dtype, _ in self.cases:
            a, b = torch.tensor(80.0, dtype=dtype), torch.tensor(60.0, dtype=dtype)
            down, up = sampling.get_ancestral_step(a, b, eta=0.0)
            self.assertIs(down, b)
            self.assertEqual(up, 0.0)
            self.assertIsInstance(up, float)

    def test_terminal_and_equal_noise_levels(self):
        for dtype, scales in self.cases:
            for scale in scales:
                for eta in (0.25, 1.0, 2.0):
                    a = torch.tensor(80 * scale, dtype=dtype)
                    for b in (torch.zeros_like(a), a.clone()):
                        with self.subTest(dtype=dtype, scale=scale, eta=eta, target=b.item()):
                            down, up = sampling.get_ancestral_step(a, b, eta)
                            torch.testing.assert_close(down, b)
                            torch.testing.assert_close(up, torch.zeros_like(b))

    def test_adjacent_noise_levels(self):
        for dtype in (torch.float32, torch.float64):
            a = torch.tensor(80.0, dtype=dtype)
            b = torch.nextafter(a, torch.zeros_like(a))
            for eta in (0.25, 1.0, 2.0):
                with self.subTest(dtype=dtype, eta=eta):
                    down, up = sampling.get_ancestral_step(a, b, eta)
                    expected = self.reference(a, b, eta)
                    actual = torch.stack((down / b, up / b))
                    torch.testing.assert_close(
                        actual, actual.new_tensor(expected),
                        rtol=self.tolerance(dtype), atol=self.tolerance(dtype),
                    )

    def test_noise_level_gradients_keep_their_analytic_scaling(self):
        for scale in (1.0, 1e100, 1e-100):
            with self.subTest(scale=scale):
                a = torch.tensor(80 * scale, dtype=torch.float64, requires_grad=True)
                b = torch.tensor(60 * scale, dtype=torch.float64, requires_grad=True)
                down, up = sampling.get_ancestral_step(a, b)
                down_grad = torch.autograd.grad(down, (a, b), retain_graph=True)
                up_grad = torch.autograd.grad(up, (a, b))
                ratio = 0.75
                root = math.sqrt(1 - ratio * ratio)
                expected_down = (-ratio * ratio, 2 * ratio)
                expected_up = (ratio ** 3 / root, (1 - 2 * ratio * ratio) / root)
                for actual, expected in zip(down_grad + up_grad, expected_down + expected_up):
                    torch.testing.assert_close(actual, actual.new_tensor(expected), rtol=1e-12, atol=1e-12)

    def test_tensor_eta_retains_its_gradient(self):
        a = torch.tensor(80.0, dtype=torch.float64)
        b = torch.tensor(60.0, dtype=torch.float64)
        eta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        down, up = sampling.get_ancestral_step(a, b, eta)
        down_grad = torch.autograd.grad(down, eta, retain_graph=True)[0]
        up_grad = torch.autograd.grad(up, eta)[0]
        torch.testing.assert_close(down_grad, eta.new_tensor(-35.0))
        torch.testing.assert_close(up_grad, eta.new_tensor(60 * math.sqrt(7) / 4))

    def test_near_one_eta_retains_the_small_half_precision_down_step(self):
        for sigma_from, sigma_to in ((80.0, 0.008), (10000.0, 1.0)):
            a = torch.tensor(sigma_from, dtype=torch.float16)
            b = torch.tensor(sigma_to, dtype=torch.float16)
            for eta in (1 - 1e-8, 1 - 1e-10):
                with self.subTest(sigma_from=sigma_from, sigma_to=sigma_to, eta=eta):
                    down, up = sampling.get_ancestral_step(a, b, eta)
                    down_ratio, up_ratio = self.reference(a, b, eta)
                    expected = down.new_tensor(down_ratio * float(b))
                    self.assertGreater(expected.item(), 0)
                    self.assertGreater(down.item(), 0)
                    # Subnormal half outputs have an absolute quantization step.
                    quantum = torch.finfo(down.dtype).tiny * torch.finfo(down.dtype).eps
                    torch.testing.assert_close(down, expected, rtol=4e-3, atol=0.51 * quantum)
                    torch.testing.assert_close(up, up.new_tensor(up_ratio * float(b)))
                    self.assertEqual((down.dtype, up.dtype), (b.dtype, b.dtype))

    def test_tensor_eta_keeps_tiny_down_values_and_finite_derivatives(self):
        for dtype, sigma_from, sigma_to in (
            (torch.float16, 80.0, 0.008),
            (torch.float16, 10000.0, 1.0),
            (torch.bfloat16, 1e30, 1e5),
            (torch.float32, 1e30, 1e5),
            (torch.float64, 1e300, 1e100),
        ):
            with self.subTest(dtype=dtype, sigma_from=sigma_from, sigma_to=sigma_to):
                a = torch.tensor(sigma_from, dtype=dtype)
                b = torch.tensor(sigma_to, dtype=dtype)
                eta = torch.tensor(1.0, dtype=dtype, requires_grad=True)
                down, up = sampling.get_ancestral_step(a, b, eta)
                down_ratio, up_ratio = self.reference(a, b, eta)
                quantum = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
                expected_down = down.new_tensor(down_ratio * float(b))
                self.assertGreater(expected_down.item(), 0)
                self.assertGreater(down.item(), 0)
                torch.testing.assert_close(
                    down, expected_down, rtol=self.tolerance(dtype), atol=0.51 * quantum,
                )
                torch.testing.assert_close(up, up.new_tensor(up_ratio * float(b)))
                derivative = torch.autograd.grad(down, eta)[0]
                expected_derivative = -float(a) * (1 - (float(b) / float(a)) ** 2)
                self.assertTrue(torch.isfinite(derivative))
                torch.testing.assert_close(
                    derivative, eta.new_tensor(expected_derivative),
                    rtol=self.tolerance(dtype), atol=0,
                )
                self.assertEqual((down.dtype, up.dtype), (dtype, dtype))

    def test_mixed_scalar_dtypes_and_clipping_keep_the_original_api(self):
        # Ordinary levels avoid the numerical defect in the baseline, so these
        # controls preserve its actual promotion and clipping types.
        for dtype in (torch.float16, torch.float32):
            for requires_grad in (False, True):
                for eta_value in (0.5, 1.0, 2.0):
                    with self.subTest(dtype=dtype, requires_grad=requires_grad, eta=eta_value):
                        a = torch.tensor(2.0, dtype=dtype)
                        b = torch.tensor(1.5, dtype=dtype)
                        eta = torch.tensor(eta_value, dtype=torch.float64, requires_grad=requires_grad)
                        down, up = sampling.get_ancestral_step(a, b, eta)
                        expected_dtype = torch.float64 if eta_value <= 1 else dtype
                        self.assertEqual((down.dtype, up.dtype), (expected_dtype, expected_dtype))
                        self.assertEqual((down.requires_grad, up.requires_grad),
                                         (requires_grad, requires_grad) if eta_value <= 1 else (False, False))
                        if eta_value > 1:
                            self.assertIs(up, b)
                            self.assertEqual(down.item(), 0.0)
            a = torch.tensor(2.0, dtype=dtype)
            for eta in (0.5, 1.0):
                down, up = sampling.get_ancestral_step(a, 1.5, eta)
                self.assertEqual((down.dtype, up.dtype), (dtype, dtype))
            down, up = sampling.get_ancestral_step(a, 1.5, 2.0)
            self.assertIsInstance(down, float)
            self.assertIsInstance(up, float)
            self.assertEqual((down, up), (0.0, 1.5))
        down, up = sampling.get_ancestral_step(2, 1, 2.0)
        self.assertIsInstance(down, float)
        self.assertIsInstance(up, int)
        self.assertEqual((down, up), (0.0, 1))
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            eta = torch.tensor(1.0, dtype=dtype)
            down, up = sampling.get_ancestral_step(2.0, 1.5, eta)
            self.assertEqual((down.dtype, up.dtype), (dtype, dtype))
            self.assertEqual((down.device, up.device), (eta.device, eta.device))
            self.assertEqual((down.requires_grad, up.requires_grad), (False, False))
            torch.testing.assert_close(down, eta.new_tensor(1.125), rtol=0, atol=0)

    def test_small_tensor_eta_keeps_existing_nonzero_gradients(self):
        for dtype, eta_value in (
            (torch.float32, 1e-8),
            (torch.float64, 1e-20),
            (torch.float64, 1e-300),
        ):
            with self.subTest(dtype=dtype, eta=eta_value):
                a = torch.tensor(2.0, dtype=dtype)
                b = torch.tensor(1.5, dtype=dtype)
                eta = torch.tensor(eta_value, dtype=dtype, requires_grad=True)
                down, _ = sampling.get_ancestral_step(a, b, eta)
                derivative = torch.autograd.grad(down, eta)[0]
                variance_ratio = 7 / 16
                expected = -float(b) * float(eta) * variance_ratio
                self.assertLess(derivative.item(), 0)
                torch.testing.assert_close(derivative, eta.new_tensor(expected),
                                           rtol=self.tolerance(dtype), atol=0)

    def test_tensor_source_and_small_python_target_keep_scalar_outputs(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                a = torch.tensor(80.0, dtype=dtype)
                b = 0.008
                down, up = sampling.get_ancestral_step(a, b)
                down_ratio, up_ratio = self.reference(a, b, 1.0)
                expected_down, expected_up = down_ratio * b, up_ratio * b
                self.assertGreater(float(down), 0)
                quantum = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
                self.assertLessEqual(abs(float(down) - expected_down),
                                     abs(expected_down) * self.tolerance(dtype) + 0.51 * quantum)
                self.assertLessEqual(abs(float(up) - expected_up), b * self.tolerance(dtype))
                # A rounded full up ratio times a Python target is a Python
                # scalar. Restoring Tensor precision must leave it as such.
                if dtype in (torch.float16, torch.bfloat16, torch.float32):
                    self.assertIsInstance(up, float)
                else:
                    self.assertEqual(up.dtype, dtype)
        for requires_grad in (False, True):
            with self.subTest(python_sigmas=True, requires_grad=requires_grad):
                eta = torch.tensor(1.0, dtype=torch.float32, requires_grad=requires_grad)
                down, up = sampling.get_ancestral_step(80.0, 0.008, eta)
                self.assertGreater(float(down), 0)
                self.assertEqual(down.dtype, eta.dtype)
                self.assertEqual(down.requires_grad, requires_grad)
                self.assertIsInstance(up, float)
                torch.testing.assert_close(down, eta.new_tensor(8e-7), rtol=3e-6, atol=0)
                if requires_grad:
                    derivative = torch.autograd.grad(down, eta)[0]
                    torch.testing.assert_close(derivative, eta.new_tensor(-80 * (1 - 1e-8)),
                                               rtol=3e-6, atol=0)

    def test_ancestral_samplers_preserve_scale_equivariance(self):
        methods = (
            sampling.sample_euler_ancestral,
            sampling.sample_dpm_2_ancestral,
            sampling.sample_dpmpp_2s_ancestral,
        )
        # A zero denoiser has the exactly solvable drift dx/dsigma = x/sigma.
        # Fixed noise makes the one-step mean/noise decomposition testable.
        for method in methods:
            for dtype, scales in (
                (torch.float16, (1.0,)),
                (torch.float32, (1.0, 1e10, 1e-14)),
                (torch.float64, (1.0, 1e100, 1e-100)),
            ):
                for scale in scales:
                    with self.subTest(method=method.__name__, dtype=dtype, scale=scale):
                        x = torch.tensor([[20.0, -10.0], [7.0, 4.0]], dtype=dtype) * scale
                        noise = x.new_tensor([[0.25, -0.5], [1.0, -0.25]])
                        sigmas = x.new_tensor([80 * scale, 60 * scale])
                        output = method(
                            lambda value, sigma: torch.zeros_like(value), x, sigmas,
                            noise_sampler=lambda sigma, sigma_next: noise, disable=True,
                        )
                        expected = x / sigmas[0] * 0.75 + noise * (math.sqrt(7) / 4)
                        torch.testing.assert_close(
                            output / sigmas[1], expected,
                            rtol=0.02 if dtype == torch.float16 else 1e-5,
                            atol=0.01 if dtype == torch.float16 else 1e-5,
                        )


if __name__ == '__main__':
    unittest.main()
