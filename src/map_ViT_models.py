"""
Patch-to-Patch Vision Transformer Models
----------------------------------------

This module implements two variants of Vision Transformer (ViT) architectures 
for dense regression tasks, designed around a patch-to-patch learning paradigm. 
Both models decompose images into non-overlapping patches, process them through 
a transformer encoder, and then reconstruct the output image.

Components:
    - Patchify:
        Splits an input image into flattened non-overlapping patches.

    - Unpatchify:
        Reconstructs an image tensor from a sequence of flattened patches.

    - ViT_Patch2Patch (Version 1):
        • Patchify + linear projection into embedding space.
        • Sinusoidal positional encoding.
        • Transformer encoder with configurable depth/heads.
        • Linear decoder back to patch space, then Unpatchify.

    - ViT_Patch2Patch_ver2 (Version 2):
        • Patch embedding via Conv2d (stride = patch size).
        • Learned positional embeddings with dropout.
        • Transformer encoder with configurable depth/heads.
        • CNN-based decoder with PixelShuffle layers for super-resolution-style
          upsampling back to the original image resolution.

Utilities:
    - test_model:
        Simple wrapper to run a forward pass and log output shapes.
    - main:
        Runs lightweight tests of both model variants on a dummy input tensor.

Usage Example:
    >>> import torch
    >>> from vit_patch2patch import ViT_Patch2Patch, ViT_Patch2Patch_ver2
    >>> model = ViT_Patch2Patch(img_size=512, patch_size=8, in_ch=3, out_ch=3)
    >>> dummy = torch.randn(1, 3, 512, 512)
    >>> out = model(dummy)
    >>> print(out.shape)   # torch.Size([1, 3, 512, 512])

Notes:
    - Logging is used to track initialization parameters and parameter counts.
    - Default settings assume square images (H = W = img_size).
    - PixelShuffle decoder in Version 2 assumes patch_size divisible by upscaling factors.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class Patchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__} | patch_size = {self.patch_size}")

    def forward(self, x):  # (B, C, H, W)
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0
        x = x.unfold(2, p, p).unfold(3, p, p)  # B, C, H//p, W//p, p, p
        x = x.permute(0, 2, 3, 1, 4, 5).flatten(1, 3)  # B, N, C, p, p
        return x.reshape(B, -1, C * p * p)  # B, N, patch_dim


class Unpatchify(nn.Module):
    def __init__(self, patch_size, out_channels, image_size):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.image_size = image_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__} | patch_size = {self.patch_size} | out_channels={out_channels} | image_size={image_size}")

    def forward(self, x):  # (B, N, patch_dim)
        B, N, D = x.shape
        p = self.patch_size
        H, W = self.image_size
        C = self.out_channels

        x = x.reshape(B, H // p, W // p, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return x


# class ViT_Patch2Patch(nn.Module):
#     def __init__(self, img_size=512, patch_size=8, in_ch=3, out_ch=3, embed_dim=512, depth=6, heads=8):
#         super().__init__()
#         self.patch_size = patch_size
#         self.img_size = img_size
#         self.num_patches = (img_size // patch_size) ** 2
#         self.patch_dim = in_ch * patch_size * patch_size
#         self.output_dim = out_ch * patch_size * patch_size
#         self.logger = logging.getLogger(self.__class__.__name__)
#         self.logger.info(f"Initalizing {self.__class__.__name__} | img_size={img_size} | patch_size={patch_size}" \
#                         "| in_ch={in_ch} | out_ch={out_ch} | embed_dim={embed_dim} | depth={depth}  | heads={heads}")
#         # Modules                
#         self.patchify = Patchify(patch_size)
#         self.proj = nn.Linear(self.patch_dim, embed_dim)
#         self.pos_encoding = PositionalEncoding(emb_size=embed_dim, max_len=self.num_patches)
#         encoder_layer = nn.TransformerEncoderLayer(embed_dim, heads, dim_feedforward=embed_dim * 4, batch_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
#         self.decoder = nn.Linear(embed_dim, self.output_dim)
#         self.unpatchify = Unpatchify(patch_size, out_ch, (img_size, img_size))
#         self._log_parameter_count()

#     def _log_parameter_count(self):
#         """
#         Logs total and trainable parameters in the model, summarized by top-level modules.
#         """
#         self.logger.info(f"{self.__class__.__name__} Parameter Summary (Top-Level Modules):")
#         self.logger.info("-" * 80)
#         total_params = 0
#         trainable_params = 0

#         for name, module in self.named_children():  # Only top-level children
#             mod_total = sum(p.numel() for p in module.parameters())
#             mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
#             total_params += mod_total
#             trainable_params += mod_trainable
#             self.logger.info(f"{name:<25} | Total: {mod_total:<20} | Trainable: {mod_trainable:,}")

#         self.logger.info("-" * 80)
#         self.logger.info(f"Total Parameters:     {total_params:,}")
#         self.logger.info(f"Trainable Parameters: {trainable_params:,}")

#     def forward(self, x):
#         x = self.patchify(x)           # (B, N, patch_dim)
#         x = self.proj(x)               # (B, N, embed_dim)
#         x = self.pos_encoding(x)       # (B, N, embed_dim)
#         x = self.encoder(x)            # (B, N, embed_dim)
#         x = self.decoder(x)            # (B, N, patch_output_dim)
#         x = self.unpatchify(x)         # (B, out_ch, H, W)
#         return x



class ViT_Patch2Patch_ver2(nn.Module):
    """ 
    SOme changes from above:
        - learned patch embed using a conv layer with kernelsize=patchsize
        - learned positional embedinng, no longer using sinusoidal 
        - added some dropout
        - decoder:
            - replaced simple linear decoder from embed dim to output dim (pre patchify)
            - using PixelShuffle super resolution technique
                o https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html 
    """
    def __init__(self, img_size=512, patch_size=8, in_ch=3, out_ch=3, embed_dim=512, depth=6, heads=8, dropout=0.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__} with img_size={img_size}, patch_size={patch_size}")

        # Patch embedding via conv
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_dropout = nn.Dropout(p=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads,
                                                   dim_feedforward=embed_dim * 4,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Decoder: maps transformer output back to full-resolution image
        self.decoder = nn.Sequential(                   # IF patchsize = 16
            nn.Conv2d(embed_dim, 512, 3, padding=1),    # [B, 768, 32, 32]  -> [B, 512, 32, 32]
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, padding=1),         # [B, 512, 32, 32]  -> [B, 1024, 32, 32]
            nn.ReLU(),
            nn.PixelShuffle(4),                         # [B, 1024, 32, 32] -> [B, 64, 128, 128]
            nn.Conv2d(64, 64, 3, padding=1),            # [B, 64, 128, 128] -> [B, 64, 128, 128]
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),            # [B, 64, 128, 128] -> [B, 64, 128, 128]
            nn.ReLU(),
            nn.PixelShuffle(4),                         # [B, 64, 128, 128] -> [B, 4, 512, 512]
            nn.Conv2d(4, out_ch, 3, padding=1)          # [B, 4, 512, 512]  -> [B, 3, 512, 512]
        )

        self._log_parameter_count()

    def _log_parameter_count(self):
        """
        Logs total and trainable parameters in the model, summarized by top-level modules.
        """
        self.logger.info(f"{self.__class__.__name__} Parameter Summary (Top-Level Modules):")
        self.logger.info("-" * 80)
        total_params = 0
        trainable_params = 0

        for name, module in self.named_children():  # Only top-level children
            mod_total = sum(p.numel() for p in module.parameters())
            mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += mod_total
            trainable_params += mod_trainable
            self.logger.info(f"{name:<25} | Total: {mod_total:<20} | Trainable: {mod_trainable:,}")

        self.logger.info("-" * 80)
        self.logger.info(f"Total Parameters:     {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")

    def forward(self, x):
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H//p, W//p]
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_dropout(x)

        # Transformer
        x = self.encoder(x)  # [B, N, embed_dim]

        # Reshape back to 2D grid
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H_p, W_p)  # [B, embed_dim, H//p, W//p]

        # Decode to full-res output
        out = self.decoder(x)  # [B, out_ch, H, W]
        return out


class ViT_Patch2Patch_ver3(nn.Module):
    """ 
    SOme changes from above:
        - learned patch embed using a conv layer with kernelsize=patchsize
        - learned positional embedinng, no longer using sinusoidal 
        - added some dropout
        - decoder:
            - replaced simple linear decoder from embed dim to output dim (pre patchify)
            - using PixelShuffle super resolution technique
                o https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html 
    """
    def __init__(self, img_size=512, patch_size=8, in_ch=3, out_ch=3, embed_dim=512, depth=6, heads=8, dropout=0.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__} with img_size={img_size}, patch_size={patch_size}")

        # Patch embedding via conv
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_dropout = nn.Dropout(p=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads,
                                                   dim_feedforward=embed_dim * 4,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Decoder: maps transformer output back to full-resolution image
        self.decoder = nn.Sequential(                   # IF patchsize = 16
            nn.Conv2d(embed_dim, 512, 3, padding=1),    # [B, 768, 32, 32]  -> [B, 512, 32, 32]
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, padding=1),         # [B, 512, 32, 32]  -> [B, 1024, 32, 32]
            nn.ReLU(),
            nn.PixelShuffle(4),                         # [B, 1024, 32, 32] -> [B, 64, 128, 128]
            nn.Conv2d(64, 64, 3, padding=1),            # [B, 64, 128, 128] -> [B, 64, 128, 128]
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),            # [B, 64, 128, 128] -> [B, 64, 128, 128]
            nn.ReLU(),
            nn.PixelShuffle(4),                         # [B, 64, 128, 128] -> [B, 4, 512, 512]
            nn.Conv2d(4, out_ch, 3, padding=1)          # [B, 4, 512, 512]  -> [B, 3, 512, 512]
        )

        self._log_parameter_count()

    def _log_parameter_count(self):
        """
        Logs total and trainable parameters in the model, summarized by top-level modules.
        """
        self.logger.info(f"{self.__class__.__name__} Parameter Summary (Top-Level Modules):")
        self.logger.info("-" * 80)
        total_params = 0
        trainable_params = 0

        for name, module in self.named_children():  # Only top-level children
            mod_total = sum(p.numel() for p in module.parameters())
            mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += mod_total
            trainable_params += mod_trainable
            self.logger.info(f"{name:<25} | Total: {mod_total:<20} | Trainable: {mod_trainable:,}")

        self.logger.info("-" * 80)
        self.logger.info(f"Total Parameters:     {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")

    def _pad_to_multiple(self, x: torch.Tensor, m: int):
        B, C, H, W = x.shape
        H2 = (H + m - 1) // m * m
        W2 = (W + m - 1) // m * m
        pad_h = H2 - H
        pad_w = W2 - W
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, H, W)
        # pad = (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (0, pad_w, 0, pad_h), (H, W)

    def _interpolate_pos_embed(self, Hp: int, Wp: int) -> torch.Tensor:
        """
        Returns pos_embed resized to (Hp*Wp, D), shape (1, Hp*Wp, D).
        Assumes stored pos_embed corresponds to a square base grid.
        """
        pos = self.pos_embed  # (1, N0, D)
        N0, D = pos.shape[1], pos.shape[2]
        Hp0 = int((N0) ** 0.5)
        Wp0 = Hp0
        if Hp0 * Wp0 != N0:
            raise ValueError(f"pos_embed length {N0} is not a perfect square; can't grid-reshape safely.")

        if (Hp, Wp) == (Hp0, Wp0):
            return pos

        pos2d = pos.reshape(1, Hp0, Wp0, D).permute(0, 3, 1, 2)  # (1, D, Hp0, Wp0)
        pos2d = F.interpolate(pos2d, size=(Hp, Wp), mode="bicubic", align_corners=False)
        pos_new = pos2d.permute(0, 2, 3, 1).reshape(1, Hp * Wp, D)  # (1, Hp*Wp, D)
        return pos_new

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) pad input so H,W divisible by patch_size
        x, pad, orig_hw = self._pad_to_multiple(x, self.patch_size)
        H0, W0 = orig_hw

        # 2) patch embed
        x = self.patch_embed(x)  # (B, D, Hp, Wp)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, D) where N=Hp*Wp

        # 3) add resized positional embedding
        pos = self._interpolate_pos_embed(Hp, Wp)  # (1, N, D)
        x = x + pos
        x = self.pos_dropout(x)

        # 4) transformer
        x = self.encoder(x)  # (B, N, D)

        # 5) reshape back to grid
        x = x.transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp)

        # 6) decode
        out = self.decoder(x)

        # 7) if we padded, crop back to original size
        out = out[..., :H0, :W0]
        return out




# ============================================================================================================
# TESTING
# ============================================================================================================

def test_model(model, name, input_tensor):
    try:
        print(f"Testing {name}...")
        out = model(input_tensor)
        print(f"{name} output shape: {out.shape}\n")
    except Exception as e:
        print(f"{name} failed with error: {e}\n")


def main():

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    
    print("Starting model tests...\n")
    # Set image parameters
    B, C, H, W = 1, 3, 512, 512
    dummy_input = torch.randn(B, C, H, W)

    # # 1. ViT Patch2Patch (version 1)
    # model1 = ViT_Patch2Patch(img_size=512, patch_size=8, in_ch=3, out_ch=3)
    # test_model(model1, "ViT_Patch2Patch (version 1)", dummy_input)
    # print("==" * 50)

    # 2. ViT Patch2Patch (version 2)
    model1 = ViT_Patch2Patch_ver2(img_size=512, patch_size=8, in_ch=3, out_ch=3)
    test_model(model1, "ViT_Patch2Patch (version 2)", dummy_input)
    print("==" * 50)




if __name__ == "__main__":
    main()
