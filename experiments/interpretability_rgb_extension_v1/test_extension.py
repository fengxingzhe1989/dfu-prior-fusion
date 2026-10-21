"""Data-free checks for the RGB Grad-CAM extension, including invalid CAMs."""
import unittest
import torch
from experiments.interpretability_rgb_extension_v1 import run


class TinyRGB(torch.nn.Module):
    def __init__(self, zero=False):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU())
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(4, 4)
        if zero:
            torch.nn.init.zeros_(self.head.weight)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


class TestExtension(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(run.MODELS, ('efficientnetb0', 'convnext_tiny', 'efficientnetv2s'))
        self.assertEqual(len(run.SEEDS)*5*len(run.MODELS), 75)

    def test_cam(self):
        torch.manual_seed(0)
        model = TinyRGB().eval()
        item = {'image': torch.randn(3, 16, 16)}
        cam, valid, probs, shape = run.gradcam(model, item, 1, torch.device('cpu'))
        self.assertEqual(cam.shape, (224, 224)); self.assertEqual(shape, [1, 4, 16, 16])
        self.assertTrue(0 <= cam.min() <= cam.max() <= 1)
        self.assertAlmostEqual(float(probs.sum()), 1, places=6)
        self.assertEqual(len(model.features[-1]._forward_hooks), 0)

    def test_zero_cam_is_unavailable(self):
        model = TinyRGB(zero=True).eval()
        cam, valid, _, _ = run.gradcam(model, {'image': torch.ones(3, 16, 16)}, 0, torch.device('cpu'))
        self.assertFalse(valid); self.assertEqual(float(cam.sum()), 0)


if __name__ == '__main__':
    unittest.main()
