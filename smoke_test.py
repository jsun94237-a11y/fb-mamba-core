"""Run model-only checks with synthetic tensors on CPU."""

from pathlib import Path
import tempfile
import unittest

import torch

from fb_mamba import FBMambaEncoder


class ModelSmokeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    @staticmethod
    def make_model():
        return FBMambaEncoder(
            input_channels=4,
            d_model=256,
            d_state=16,
            n_layers=3,
            dropout=0.1,
        )

    def test_forward_variable_lengths(self):
        model = self.make_model().eval()
        with torch.no_grad():
            for length in (1, 17, 32, 33):
                with self.subTest(length=length):
                    outputs = model(torch.randn(2, length, 4))
                    self.assertEqual(tuple(outputs.shape), (2, 256))
                    self.assertTrue(torch.isfinite(outputs).all().item())

    def test_backward_and_optimizer_step(self):
        model = self.make_model().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.randn(2, 32, 4, requires_grad=True)
        targets = torch.randn(2, 256)
        before = model.input_proj.weight.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all().item())
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())

        optimizer.step()
        self.assertFalse(torch.equal(before, model.input_proj.weight))
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter).all().item())

    def test_checkpoint_round_trip(self):
        model = self.make_model().eval()
        inputs = torch.randn(2, 17, 4)
        with torch.no_grad():
            expected = model(inputs)

        with tempfile.TemporaryDirectory(prefix="fb_mamba_smoke_") as directory:
            checkpoint = Path(directory) / "encoder.pt"
            torch.save(model.state_dict(), checkpoint)
            restored = self.make_model().eval()
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            restored.load_state_dict(state, strict=True)
            with torch.no_grad():
                actual = restored(inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
