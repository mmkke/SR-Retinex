'''
ResNet-based U-Net Models (with Optional ViT Bottleneck)
--------------------------------------------------------

This module implements U-Net-style decoders on top of ResNet encoders, with
optional Squeeze-and-Excitation (SE) attention and a hybrid variant that inserts
a Transformer (ViT-like) bottleneck. Designed for dense prediction/regression
tasks on images (e.g., intrinsic imaging, illumination/ISD estimation).

Components
----------
- SEBlock:
    Channel-wise attention (Squeeze-and-Excitation) to recalibrate feature maps.

- UpBlock:
    Decoder block with bilinear upsampling, skip concatenation, Conv-BN-ReLU
    stacks, optional dropout, and optional SEBlock.

Models
------
- ResNet50UNet:
    ResNet-50 encoder (conv1..layer4) with a 5-stage decoder. Supports:
      • pretrained or checkpointed weights
      • optional SE blocks in decoder
      • arbitrary input channels (replaces conv1 when in_channels != 3)

- ResNet18UNet:
    Lighter variant using ResNet-18 encoder with an analogous decoder and options.

- ResNet50UNet_ViT:
    Hybrid model that adds an optional Transformer bottleneck:
      • 1×1 projection (2048 → vit_embed_dim)
      • learned positional embeddings for a fixed token grid
      • TransformerEncoder (depth, heads, ff-dim configurable)
      • 1×1 unprojection (vit_embed_dim → 2048)
    Also supports freezing encoder and/or decoder for transfer learning.
'''

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


# ============================================================================================================
# ResNet UNet
# ============================================================================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale.to(dtype=x.dtype, device=x.device)

################################################################################################################################################################

class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, se_block=False, dropout=0.0):
        super().__init__()
        self.se_block=se_block
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        if se_block:
            self.se = SEBlock(out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        if self.se_block:
            return self.se(x)
        return x

################################################################################################################################################################

class ResNet50UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, pretrained=True, checkpoint=None, se_block=True, dropout=0.0):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Initializing {self.__class__.__name__} | "
            f"in_channels={in_channels}, out_channels={out_channels}, "
            f"pretrained={pretrained}, checkpoint={checkpoint}, se_block={se_block}"
        )
        # Load backbone
        if checkpoint:
            self.logger.info(f"Loading ResNet50 weights from checkpoint: {checkpoint}")
            resnet = resnet50(weights=None)
            state_dict = torch.load(checkpoint, map_location="cpu")
            resnet.load_state_dict(state_dict.get("state_dict", state_dict), strict=False)
        else:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = resnet50(weights=weights)


        # Encoder
        self.in_conv = nn.Sequential(
            resnet.conv1,  # 64 x H/2
            resnet.bn1, 
            resnet.relu
        )
        self.maxpool = resnet.maxpool  # H/4
        self.enc1 = resnet.layer1  # 256 x H/4
        self.enc2 = resnet.layer2  # 512 x H/8
        self.enc3 = resnet.layer3  # 1024 x H/16
        self.enc4 = resnet.layer4  # 2048 x H/32

        # Decoder
        self.up4 = UpBlock(2048, 1024, 512, se_block=se_block, dropout=dropout)
        self.up3 = UpBlock(512, 512, 256, se_block=se_block, dropout=dropout)
        self.up2 = UpBlock(256, 256, 128, se_block=se_block, dropout=dropout)
        self.up1 = UpBlock(128, 64, 64, se_block=se_block, dropout=dropout)
        self.up0 = UpBlock(64, 0, 32, se_block=se_block, dropout=dropout)  # No skip connection here
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)


        # Optionally re-train input conv to accept different channels
        if in_channels != 3:
            self.in_conv[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        # Encoder
        self.logger.debug(f"{self.__class__.__name__} input shape = {x.shape}")
        x0 = self.in_conv(x)       # conv1
        x1 = self.enc1(self.maxpool(x0))  # layer1
        x2 = self.enc2(x1)         # layer2
        x3 = self.enc3(x2)         # layer3
        x4 = self.enc4(x3)         # layer4
        self.logger.debug(f"{self.__class__.__name__} bottlneck shape = {x4.shape}")
        # Decoder with skip connections
        d4 = self.up4(x4, x3) # skip enc layer4
        d3 = self.up3(d4, x2) # skip enc layer2`
        d2 = self.up2(d3, x1) # skip enc layer1
        d1 = self.up1(d2, x0)
        d0 = self.up0(d1, None) # no skip

        out = self.final_conv(d0)  # regression output
        self.logger.debug(f"{self.__class__.__name__} output shape = {out.shape}")

        return out


################################################################################################################################################################

class ResNet18UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, pretrained=True, checkpoint=None, se_block=True, dropout=0.0):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Initializing {self.__class__.__name__} | "
            f"in_channels={in_channels}, out_channels={out_channels}, "
            f"pretrained={pretrained}, checkpoint={checkpoint}, se_block={se_block}"
        )
        # Load backbone
        if checkpoint:
            self.logger.info(f"Loading ResNet18 weights from checkpoint: {checkpoint}")
            resnet = resnet18(weights=None)
            state_dict = torch.load(checkpoint, map_location="cpu")
            resnet.load_state_dict(state_dict.get("state_dict", state_dict), strict=False)
        else:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            resnet = resnet18(weights=weights)


        # Encoder
        self.in_conv = nn.Sequential(
            resnet.conv1,  # 64 x H/2
            resnet.bn1, 
            resnet.relu
        )
        self.maxpool = resnet.maxpool  # H/4
        self.enc1 = resnet.layer1  # 64 x H/4
        self.enc2 = resnet.layer2  # 128 x H/8
        self.enc3 = resnet.layer3  # 256 x H/16
        self.enc4 = resnet.layer4  # 512 x H/32
        
        # Decoder
        self.up4 = UpBlock(512, 256, 256, se_block=se_block, dropout=dropout)   # x4, x3 → 256
        self.up3 = UpBlock(256, 128, 128, se_block=se_block, dropout=dropout)   # x3, x2 → 128
        self.up2 = UpBlock(128, 64, 64, se_block=se_block, dropout=dropout)     # x2, x1 → 64
        self.up1 = UpBlock(64, 64, 32, se_block=se_block, dropout=dropout)      # x1, x0 → 32
        self.up0 = UpBlock(32, 0, 16, se_block=se_block, dropout=dropout)       # no skip
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)


        # Optionally re-train input conv to accept different channels
        if in_channels != 3:
            self.in_conv[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        # Encoder
        self.logger.debug(f"{self.__class__.__name__} input shape = {x.shape}")
        x0 = self.in_conv(x)       # conv1
        x1 = self.enc1(self.maxpool(x0))  # layer1
        x2 = self.enc2(x1)         # layer2
        x3 = self.enc3(x2)         # layer3
        x4 = self.enc4(x3)         # layer4
        self.logger.debug(f"{self.__class__.__name__} bottlneck shape = {x4.shape}")
        # Decoder with skip connections
        d4 = self.up4(x4, x3) # skip enc layer4
        d3 = self.up3(d4, x2) # skip enc layer2`
        d2 = self.up2(d3, x1) # skip enc layer1
        d1 = self.up1(d2, x0)
        d0 = self.up0(d1, None) # no skip

        out = self.final_conv(d0)  # regression output
        self.logger.debug(f"{self.__class__.__name__} output shape = {out.shape}")

        return out

################################################################################################################################################################

class ResNet50UNet_ViT(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, pretrained=True, checkpoint=None,
                 use_vit_bottleneck=False, vit_embed_dim=512, vit_depth=4, vit_heads=8, 
                 se_block=False, freeze_encoder=False, freeze_decoder=False):
        super().__init__()
        self.use_vit_bottleneck = use_vit_bottleneck

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Initalizing {self.__class__.__name__} |  in_channels = {in_channels} | out_channels = {out_channels}"
            f"| pretrained = {pretrained} | checkpoint={checkpoint} | use_vit_bottleneck={use_vit_bottleneck}"
            f"| se_block = {se_block} | freeze_encoder = {freeze_encoder} | freeze_decoder = {freeze_decoder}"
        )
        # Load backbone
        if checkpoint:
            self.logger.info(f"Loading ResNet50 weights from checkpoint: {checkpoint}")
            resnet = resnet50(weights=None)
            state_dict = torch.load(checkpoint, map_location="cpu")
            resnet.load_state_dict(state_dict.get("state_dict", state_dict), strict=False)
        else:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = resnet50(weights=weights)

        # Encoder
        self.in_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        # Optional ViT bottleneck
        if use_vit_bottleneck:
            self.vit_proj = nn.Conv2d(2048, vit_embed_dim, kernel_size=1)
            self.pos_embed = nn.Parameter(torch.zeros(1, 16*16, vit_embed_dim))
            encoder_layer = nn.TransformerEncoderLayer(d_model=vit_embed_dim, nhead=vit_heads,
                                                       dim_feedforward=vit_embed_dim * 4, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=vit_depth)
            self.vit_unproj = nn.Conv2d(vit_embed_dim, 2048, kernel_size=1)

        # Decoder
        self.up4 = UpBlock(2048, 1024, 512, se_block=se_block)
        self.up3 = UpBlock(512, 512, 256, se_block=se_block)
        self.up2 = UpBlock(256, 256, 128, se_block=se_block)
        self.up1 = UpBlock(128, 64, 64, se_block=se_block)
        self.up0 = UpBlock(64, 0, 32, se_block=se_block)  # No skip connection here
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        # Handle non-3 input channels
        if in_channels != 3:
            self.in_conv[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Freeze encoder if requested
        if freeze_encoder:
            encoder_modules = [
                self.in_conv, self.maxpool,
                self.enc1, self.enc2, self.enc3, self.enc4
            ]
            for module in encoder_modules:
                for param in module.parameters():
                    param.requires_grad = False
            self.logger.info("Encoder layers frozen.")

        # Freeze decoder if requested
        if freeze_decoder:
            decoder_modules = [
                self.up4, self.up3, self.up2, self.up1, self.up0,
                self.final_conv
            ]
            for module in decoder_modules:
                for param in module.parameters():
                    param.requires_grad = False
            self.logger.info("Decoder layers frozen.")

        self._log_parameter_count()

    def _log_parameter_count(self):
        self.logger.info(f"{self.__class__.__name__} Parameter Summary (Top-Level Modules):")
        self.logger.info("-" * 80)
        total_params, trainable_params = 0, 0
        for name, module in self.named_children():
            mod_total = sum(p.numel() for p in module.parameters())
            mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += mod_total
            trainable_params += mod_trainable
            self.logger.info(f"{name:<25} | Total: {mod_total:<20} | Trainable: {mod_trainable:,}")
        self.logger.info("-" * 80)
        self.logger.info(f"Total Parameters:     {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")

    def forward(self, x):
        self.logger.debug(f"{self.__class__.__name__} input shape = {x.shape}")
        x0 = self.in_conv(x)
        x1 = self.enc1(self.maxpool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        self.logger.debug(f"{self.__class__.__name__} bottleneck shape = {x4.shape}")

        # Optional ViT bottleneck
        if self.use_vit_bottleneck:
            B, C, H, W = x4.shape  # Expecting H=W=16 for 512x512 input
            x_vit = self.vit_proj(x4)                      # B, embed_dim, H, W
            x_vit = x_vit.flatten(2).transpose(1, 2)       # B, N, D
            x_vit = x_vit + self.pos_embed[:, :x_vit.size(1), :]
            x_vit = self.transformer(x_vit)                # B, N, D
            x_vit = x_vit.transpose(1, 2).reshape(B, -1, H, W)
            x4 = self.vit_unproj(x_vit)                    # B, 2048, H, W

        # Decoder
        d4 = self.up4(x4, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)
        d0 = self.up0(d1, None)
        out = self.final_conv(d0)

        self.logger.debug(f"{self.__class__.__name__} output shape = {out.shape}")
        return out
