# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Optional
import torch
import torch.nn as nn
import time
import os
import re
import json

import timm
from transformers import AutoModel, AutoConfig
from timm.models.vision_transformer import resize_pos_embed


class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: int = 16,
        backbone_name: str = "",
        ckpt_path: str = "",
        device: str = "cuda:0",
    ):
        super().__init__()

        print("==> Initializing ViT backbone ...")
        
        # Workaround: jsonargparse doesn't pass ckpt_path in nested configs
        # Try environment variable or look for checkpoint in standard location
        if not ckpt_path or ckpt_path == "":
            ckpt_path = os.environ.get("DINOV3_CHECKPOINT_PATH", "")
            if not ckpt_path:
                # Try default location
                default_path = os.path.join(os.path.dirname(__file__), "..", "model06000.ckpt")
                if os.path.exists(default_path):
                    ckpt_path = default_path
                    print(f"  Using checkpoint from default location: {ckpt_path}")
        
        print(f"  img_size={img_size}")
        print(f"  ckpt_path={ckpt_path}")

        if not ckpt_path or not os.path.exists(ckpt_path):
            raise ValueError(
                f"Checkpoint path not found. Tried: {ckpt_path}\n"
                f"Set DINOV3_CHECKPOINT_PATH environment variable or place model06000.ckpt in project root"
            )
        
        # Use HuggingFace model name to get architecture (internet access OK)
        # This downloads only the config, not the pretrained weights
        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        cfg = AutoConfig.from_pretrained(model_name)
        
        # Build model architecture (no pretrained weights loaded)
        _t0_tf2timm = time.time()
        self.backbone = self.transformers_to_timm(
            AutoModel.from_config(cfg),  # from_config doesn't load pretrained weights
            img_size,
        )
        self.backbone.to("cpu")
        print(f"timing:transformers_to_timm_s={time.time() - _t0_tf2timm:.3f}")
        
        # Load YOUR checkpoint weights (not Facebook's pretrained)
        self.load_encoder_from_ckpt(ckpt_path)

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)
        self.backbone.to(device)
    
    def load_encoder_from_ckpt(self, ckpt_path):
        print(f"==> Loading encoder weights from checkpoint: {ckpt_path}")
        _ckpt_bytes = None
        try:
            _ckpt_bytes = os.path.getsize(ckpt_path)
        except Exception:
            _ckpt_bytes = -1
        _t0_load = time.time()
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            _used_weights_only = True
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            _used_weights_only = False
        _torchload_s = time.time() - _t0_load

        # 1) get the raw state_dict (supports both styles)
        raw = ckpt.get("state_dict", ckpt)               # Lightning-style or plain
        
        # Handle nested structures
        if isinstance(raw, dict) and "network" in raw and isinstance(raw["network"], dict):
            raw = raw["network"]                         # nested dict
        
        # Check if this is a direct DINOv3 checkpoint (model.module.dinov3.*)
        if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
            raw = raw["model"]                           # DINOv3 checkpoint format
        
        # Try to isolate encoder weights
        if isinstance(raw, dict) and "encoder" in raw and isinstance(raw["encoder"], dict):
            enc_sd = raw["encoder"]                      # already isolated
        else:
            # fall back to prefix filter
            prefixes = ("network.encoder.", "model.encoder.", "encoder.", "backbone.", "module.dinov3.", "module.", "dinov3.")
            enc_sd = {}
            for k, v in raw.items():
                # Skip optimizer/scheduler keys
                if any(k.startswith(p) for p in ["optimizer", "scheduler", "step", "epoch"]):
                    continue
                
                # Try to strip known prefixes
                stripped_key = k
                for p in prefixes:
                    if k.startswith(p):
                        stripped_key = k[len(p):]
                        break
                
                # Use the stripped key
                enc_sd[stripped_key] = v

        if not enc_sd:
            raise RuntimeError("Couldn't find encoder weights in checkpoint.")

        # 2) clean up any remaining prefixes and map layer names
        cleaned = {}
        for k, v in enc_sd.items():
            # Remove common prefixes
            for prefix in ["backbone.", "module.", "dinov3."]:
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            
            # Skip keys that don't exist in HuggingFace model
            if k in ["mask_token", "rope_embed.periods"]:
                continue
            
            # Special handling for QKV weights - need to split into Q, K, V
            if "attn.qkv" in k:
                block_idx = k.split(".")[1]
                if k.endswith(".weight"):
                    # Split QKV weight into Q, K, V (shape: [3*dim, dim])
                    dim = v.shape[1]
                    q, k_w, v_w = v.chunk(3, dim=0)
                    cleaned[f"blocks.{block_idx}.attention.q_proj.weight"] = q
                    cleaned[f"blocks.{block_idx}.attention.k_proj.weight"] = k_w
                    cleaned[f"blocks.{block_idx}.attention.v_proj.weight"] = v_w
                elif k.endswith(".bias"):
                    # Split QKV bias into Q, K, V (shape: [3*dim])
                    # Note: HuggingFace DINOv3 doesn't use bias for K projection
                    dim = v.shape[0] // 3
                    q, k_b, v_b = v.chunk(3, dim=0)
                    cleaned[f"blocks.{block_idx}.attention.q_proj.bias"] = q
                    # Skip k_proj.bias - not used in HuggingFace DINOv3
                    cleaned[f"blocks.{block_idx}.attention.v_proj.bias"] = v_b
                continue
            
            # Map DINOv3 checkpoint names to HuggingFace model names
            k = self._map_layer_name(k)
            cleaned[k] = v

        # 3) handle positional embedding resize if image size / patch grid changed
        if "pos_embed" in cleaned and hasattr(self.backbone, "pos_embed"):
            pe_ckpt = cleaned["pos_embed"]
            pe_model = self.backbone.pos_embed
            if pe_ckpt.shape != pe_model.shape:
                _t0_pe = time.time()
                # num_tokens=1 for [CLS]; grid size inferred from model
                cleaned["pos_embed"] = resize_pos_embed(
                    pe_ckpt, pe_model, num_tokens=1, gs_new=getattr(self.backbone.patch_embed, "grid_size", None)
                )
                print(f"timing:pos_embed_resize_s={time.time() - _t0_pe:.3f}")

        # 4) classifier heads in ckpt can be ignored since num_classes=0 removed head
        # (timm names vary: 'head.weight', 'head.bias', sometimes 'fc.*')
        for drop_key in ("head.weight", "head.bias", "fc.weight", "fc.bias", "classifier.weight", "classifier.bias"):
            cleaned.pop(drop_key, None)

        _t0_apply = time.time()
        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        _apply_s = time.time() - _t0_apply
        print(
            f"timing:encoder_ckpt_bytes={_ckpt_bytes} weights_only={_used_weights_only} torch_load_s={_torchload_s:.3f} apply_state_s={_apply_s:.3f}"
        )
        print("[encoder load] missing:", missing)
        print("[encoder load] unexpected:", unexpected)

    def _map_layer_name(self, name: str) -> str:
        """Map DINOv3 checkpoint layer names to HuggingFace model names"""
        
        # CLS token mapping
        if name == "cls_token":
            return "patch_embed.cls_token"
        
        # Patch embedding mappings
        if name == "patch_embed.proj.weight":
            return "patch_embed.patch_embeddings.weight"
        if name == "patch_embed.proj.bias":
            return "patch_embed.patch_embeddings.bias"
        
        # Storage tokens -> register tokens
        if name == "storage_tokens":
            return "patch_embed.register_tokens"
        
        # Block-level mappings
        if "blocks." in name:
            parts = name.split(".")
            block_idx = parts[1]
            rest = ".".join(parts[2:])
            
            # Attention mappings: attn.qkv -> attention.{q,k,v}_proj
            if rest.startswith("attn.qkv"):
                # Special handling for QKV - need to split into separate q, k, v
                # For now, keep as is and handle in load
                return name
            elif rest.startswith("attn.proj"):
                return f"blocks.{block_idx}.attention.o_proj" + rest[len("attn.proj"):]
            elif rest.startswith("attn."):
                return f"blocks.{block_idx}.attention." + rest[len("attn."):]
            
            # Layer scale mappings
            elif rest == "ls1.gamma":
                return f"blocks.{block_idx}.layer_scale1.lambda1"
            elif rest == "ls2.gamma":
                return f"blocks.{block_idx}.layer_scale2.lambda1"
            
            # MLP mappings
            elif rest == "mlp.fc1.weight":
                return f"blocks.{block_idx}.mlp.up_proj.weight"
            elif rest == "mlp.fc1.bias":
                return f"blocks.{block_idx}.mlp.up_proj.bias"
            elif rest == "mlp.fc2.weight":
                return f"blocks.{block_idx}.mlp.down_proj.weight"
            elif rest == "mlp.fc2.bias":
                return f"blocks.{block_idx}.mlp.down_proj.bias"
            
            # Norm mappings (already correct format)
            elif rest.startswith("norm"):
                return name
        
        return name
    
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