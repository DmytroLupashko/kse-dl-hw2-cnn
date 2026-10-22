"""`ResnetFromPaper` vs torchvision `resnet34`. Run: `python -m unittest core.test_custom_model`."""

import unittest

import torch
import torchvision

from core.model import ResnetFromPaper

NUM_CLASSES = 5
STAGES = {  # block name: output shape for 224x224 input
    "pool1": (64, 56, 56),
    "resnetBlock64_3": (64, 56, 56),
    "resnetBlock128_4": (128, 28, 28),
    "resnetBlock256_6": (256, 14, 14),
    "resnetBlock512_3": (512, 7, 7),
}


class TestResnetFromPaper(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.ours = ResnetFromPaper(NUM_CLASSES, use_batchnorm=True).eval()
        self.torchvision = torchvision.models.resnet34(num_classes=NUM_CLASSES).eval()
        self.x = torch.randn(2, 3, 224, 224)

    @staticmethod
    def _is_conv_bias(key):
        return key.endswith("conv.bias")

    def _count(self, model, module_type):
        return sum(isinstance(m, module_type) for m in model.modules())

    def _randomize_batchnorm(self, model):
        """Non-trivial BN stats and affine, else BN is identity at init and misplaced BN goes unseen."""
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.running_mean.normal_()
                    m.running_var.uniform_(0.5, 2.0)
                    m.weight.uniform_(0.5, 1.5)
                    m.bias.normal_()

    def _copy_torchvision_weights(self):
        """Same module order in both models: copy tensors one by one; our conv biases set to zero (torchvision has none)."""
        ours_state = self.ours.state_dict()
        targets = [t for k, t in ours_state.items() if not self._is_conv_bias(k)]
        sources = list(self.torchvision.state_dict().values())
        self.assertEqual(len(targets), len(sources))
        with torch.no_grad():
            for k, t in ours_state.items():
                if self._is_conv_bias(k):
                    t.zero_()
            for source, target in zip(sources, targets):
                self.assertEqual(source.shape, target.shape)
                target.copy_(source)

    def test_output_shape(self):
        self.assertEqual(tuple(self.ours(self.x).shape), (2, NUM_CLASSES))

    def test_stage_shapes(self):
        shapes = {}
        for name in STAGES:
            getattr(self.ours, name).register_forward_hook(
                lambda module, inputs, output, name=name: shapes.__setitem__(name, tuple(output.shape[1:]))
            )
        self.ours(self.x)
        self.assertEqual(shapes, STAGES)

    def test_layer_counts(self):
        for module_type in (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear):
            self.assertEqual(self._count(self.ours, module_type), self._count(self.torchvision, module_type), module_type.__name__)

    def test_parameter_count(self):
        conv_biases = sum(t.numel() for k, t in self.ours.named_parameters() if self._is_conv_bias(k))
        ours = sum(p.numel() for p in self.ours.parameters()) - conv_biases
        self.assertEqual(ours, sum(p.numel() for p in self.torchvision.parameters()))

    def test_same_output_as_torchvision(self):
        self._randomize_batchnorm(self.torchvision)
        self._copy_torchvision_weights()
        with torch.no_grad():
            torch.testing.assert_close(self.ours(self.x), self.torchvision(self.x), rtol=1e-4, atol=1e-4)

    def test_gradients_reach_every_parameter(self):
        self.ours.train()
        self.ours(self.x).sum().backward()
        missing = [k for k, p in self.ours.named_parameters() if p.grad is None or not p.grad.abs().sum()]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
