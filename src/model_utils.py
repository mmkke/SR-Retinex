import os
import cv2
import numpy as np
import logging
from pathlib import Path

import torch

from map_ViT_models import ViT_Patch2Patch_ver3


REPO_ROOT = Path(__file__).resolve().parents[1]
        
        
def load_model(model_type="vit"): 

    if model_type == "vit":
        model = ViT_Patch2Patch_ver3(
                        img_size=512, 
                        patch_size=16, 
                        in_ch=3, 
                        out_ch=3, 
                        embed_dim=768, 
                        depth=12,
                        heads=8,
                        dropout=0
                )
        model_path = REPO_ROOT / "model" / "vit_12_linear.pth"
    elif model_type == "resnet":
        from unet_models3 import ResNet50UNet

        model = ResNet50UNet(
            in_channels=3, 
            out_channels=3, 
            pretrained=False, 
            checkpoint=None, 
            se_block=True
            )
        model_path = REPO_ROOT / "model" / "UNET_run_x10_01_extended_best_model.pth"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model, model_path

class ISDMapEstimator:
    def __init__(self, model: object, model_path: str, device: str = "cpu"):
        """
        Initializes the ISDModelPredictor with a pretrained model.

        Parameters:
        -----------
        model_path : str
            Path to the `.pth` file containing the saved model state_dict.
        device : str
            Device to run the model on ('cpu' or 'cuda').
        logger : logging.Logger
            Optional logger instance.
        """
        self.device = torch.device(device)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Device = {self.device}")
        self.sr_map = None

        # Load model
        self.model = model
        if model_path is not None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint from {model_path}: {e}")
            if "model_state_dict" not in checkpoint:
                raise KeyError(f"'model_state_dict' key not found in checkpoint at: {model_path}")

            try:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError as e:
                raise RuntimeError(f"Failed to load model state_dict: {e}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _reshape_image(self, img, size=512):
        """
        Crops the largest centered square from the input image and resizes it to `size`×`size`.

        Args:
            img (np.ndarray): Input image (H, W, C) or (H, W)
            size (int): Output resolution (default 512)

        Returns:
            np.ndarray: Cropped and resized image.
        """
        h, w = img.shape[:2]
        side = min(h, w)

        # Compute crop offsets
        top = (h - side) // 2
        left = (w - side) // 2

        # Crop the centered square
        cropped = img[top:top+side, left:left+side]

        # Resize to target size
        resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

        return resized

    def _preprocess_image(self, image: np.ndarray, log=True) -> torch.Tensor:
        """
        Converts a 16-bit linear RGB image to a normalized log-space tensor for model input.

        Parameters:
        -----------
        image : np.ndarray
            Input image in 16-bit format (H, W, 3).
        log : bool
            If True, convernt image to log space. Otherwise use scaled linear image

        Returns:
        --------
        torch.Tensor
            Normalized tensor of shape (1, 3, H, W) in log-space.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        if log:
            processed = np.zeros_like(image, dtype=np.float32)
            mask = image > 0
            processed[mask] = np.log(image[mask])

            # 16-bit max log ≈ log(65535) ≈ 11.09
            assert processed.min() >= 0
            assert processed.max() <= 11.1

            processed /= 11.1  # normalize to [0, 1]

        else:
            # assert image.min() >= 0
            # assert image.max() <= 65535

            # processed = image / 65535.0  # normalize to [0, 1]
            assert image.min() >= 0
            assert image.max() <= 1.0
            processed = image
        input_tensor = (
            torch.from_numpy(processed.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )

        self.logger.info(
            f"Input Tensor | Shape: {input_tensor.shape} | "
            f"Dtype: {input_tensor.dtype} | "
            f"Min: {input_tensor.min().item():.4f} | "
            f"Max: {input_tensor.max().item():.4f}"
        )
        return input_tensor


    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Runs inference on the input image and returns a normalized ISD map.

        Parameters:
        -----------
        image : np.ndarray
            Input image (H, W, 3), typically 16-bit RGB.

        Returns:
        --------
        np.ndarray
            Normalized ISD map of shape (H, W, 3) in float32.
        """
        # image = self._reshape_image(image)
        input_tensor = self._preprocess_image(image, log=False)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              with torch.no_grad():
            output = self.model(input_tensor)

        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        norm = np.linalg.norm(output_np, axis=2, keepdims=True).astype(np.float32)
        norm[norm == 0] = 1.0
        self.sr_map = output_np / norm

        self.logger.info(f"Map | Shape: {self.sr_map.shape} | Dtype: {self.sr_map.dtype}\n")
        return self.sr_map, image
    
    def get_pixelwise_angular_dist(self, sr_map_target):

        # Flatten to (-1, 3)
        pred_flat = self.sr_map.reshape(-1, 3)
        target_flat = sr_map_target.reshape(-1, 3)

        pred_norms = np.linalg.norm(pred_flat, ord=2, axis=1)
        target_norms = np.linalg.norm(target_flat, ord=2, axis=1)

        # Valid vectors: both pred and target must be non-zero
        valid_mask = (pred_norms > 0) & (target_norms > 0)

        if not np.any(valid_mask):
            raise ValueError("No valid vectors found to compute angular error.")

        # Normalize only valid entries
        pred_unit = pred_flat[valid_mask] / pred_norms[valid_mask, np.newaxis]
        target_unit = target_flat[valid_mask] / target_norms[valid_mask, np.newaxis]

        # Cosine similarity
        cos_sim = np.sum(pred_unit * target_unit, axis=1)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)  # ensure within valid range

        # Angular error in degrees
        angles = np.arccos(cos_sim) * (180.0 / np.pi)
        return angles.mean()
        
