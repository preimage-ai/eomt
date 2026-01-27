# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import timm
from transformers import AutoModel


class ViTMultiScale(nn.Module):
    """
    Multi-scale ViT encoder that can handle variable input sizes.
    Uses interpolated position embeddings to support both (512, 512) and (512, 1024).
    """
    def __init__(
        self,
        img_size: Union[Tuple[int, int], list] = (512, 1024),
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        ckpt_path: Optional[str] = None,
        support_multi_resolution: bool = True,
    ):
        super().__init__()
        
        self.support_multi_resolution = support_multi_resolution
        self.base_img_size = img_size if isinstance(img_size, tuple) else tuple(img_size)

        if "/" in backbone_name:
            self.backbone = self.transformers_to_timm(
                AutoModel.from_pretrained(backbone_name),
                img_size,
            )
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=ckpt_path is None,
                img_size=img_size,
                patch_size=patch_size,
                num_classes=0,
            )

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)
        
        # Store original position embeddings for interpolation
        if hasattr(self.backbone, 'pos_embed'):
            self.register_buffer("base_pos_embed", self.backbone.pos_embed.clone())

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int):
        """
        Interpolate position embeddings to match input size.
        This allows the model to handle variable input resolutions.
        """
        if not self.support_multi_resolution:
            return
            
        npatch = x.shape[1] - self.backbone.num_prefix_tokens
        N = self.base_pos_embed.shape[1] - self.backbone.num_prefix_tokens
        
        if npatch == N and w == h:
            return
        
        # Separate class token and position embeddings
        class_pos_embed = self.base_pos_embed[:, :self.backbone.num_prefix_tokens]
        patch_pos_embed = self.base_pos_embed[:, self.backbone.num_prefix_tokens:]
        
        dim = x.shape[-1]
        w0 = w // self.backbone.patch_embed.patch_size[1]
        h0 = h // self.backbone.patch_embed.patch_size[0]
        
        # Interpolate patch embeddings
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(N**0.5), int(N**0.5), dim).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode='bicubic',
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        
        # Concatenate class token and interpolated patch embeddings
        self.backbone.pos_embed = nn.Parameter(
            torch.cat((class_pos_embed, patch_pos_embed), dim=1)
        )

    def transformers_to_timm(self, backbone, img_size: tuple[int, int]):
        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )

        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.patch_embed.config.num_register_tokens + 1
        backbone.blocks = backbone.layer

        del (
            backbone.patch_embed.mask_token,
            backbone.embeddings,
            backbone.layer,
        )

        return backbone

    def forward(self, x: torch.Tensor):
        """
        Forward pass with dynamic position embedding interpolation.
        """
        B, C, H, W = x.shape
        
        # Interpolate position embeddings if needed
        if self.support_multi_resolution and hasattr(self.backbone, 'pos_embed'):
            self.interpolate_pos_encoding(x, W, H)
        
        return self.backbone(x)
