import os
import sys
import unittest
from pathlib import Path
from PIL import Image
import torch

# Ensure clothing path is in sys.path
clothing_dir = Path(__file__).resolve().parent.parent
if str(clothing_dir) not in sys.path:
    sys.path.insert(0, str(clothing_dir))

from train_classifier import get_classifier_transforms, PyTorchImageFolderDataset, TowelStateDataset


class TestClassifierTransforms(unittest.TestCase):
    def setUp(self):
        self.dummy_img = Image.new("RGB", (640, 480), color=(128, 128, 128))

    def test_get_classifier_transforms_train(self):
        transform_train = get_classifier_transforms(is_train=True, augment=True)
        img_t = transform_train(self.dummy_img)
        self.assertIsInstance(img_t, torch.Tensor)
        self.assertEqual(img_t.shape, (3, 224, 224))

    def test_get_classifier_transforms_eval(self):
        transform_eval = get_classifier_transforms(is_train=False, augment=False)
        img_t1 = transform_eval(self.dummy_img)
        img_t2 = transform_eval(self.dummy_img)
        self.assertIsInstance(img_t1, torch.Tensor)
        self.assertEqual(img_t1.shape, (3, 224, 224))
        # Eval transform must be deterministic
        self.assertTrue(torch.allclose(img_t1, img_t2))

    def test_pytorch_image_folder_dataset_augmentation_flags(self):
        dataset_train = PyTorchImageFolderDataset(
            dataset_dir=clothing_dir / "non_existent_folder",
            is_train=True,
            augment=True,
        )
        dataset_val = PyTorchImageFolderDataset(
            dataset_dir=clothing_dir / "non_existent_folder",
            is_train=False,
            augment=False,
        )
        self.assertIsNotNone(dataset_train.transform)
        self.assertIsNotNone(dataset_val.transform)


if __name__ == "__main__":
    unittest.main()
