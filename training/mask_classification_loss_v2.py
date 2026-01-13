# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
import json
from pathlib import Path
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)


class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        class_weights_path: Optional[str] = None,
        ade_asymmetric_loss: bool = False,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        self.ade_asymmetric_loss = ade_asymmetric_loss
        
        # Load class weights and rooftop class IDs
        class_weights, rooftop_class_ids = self._load_class_weights(class_weights_path, num_labels)
        self.rooftop_class_ids = rooftop_class_ids  # classes only annotated in rooftop samples
        
        empty_weight = torch.ones(self.num_labels + 1)
        if class_weights is not None:
            empty_weight[:self.num_labels] = class_weights
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        
        # Create weight tensor for ADE20K samples (rooftop classes masked to 0)
        ade20k_weight = empty_weight.clone()
        for cid in self.rooftop_class_ids:
            if 0 <= cid < num_labels:
                ade20k_weight[cid] = 0.0
        self.register_buffer("ade20k_weight", ade20k_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )
    
    def _load_class_weights(self, weights_path: Optional[str], num_labels: int) -> tuple:
        """
        Load class weights from JSON file.
        
        IMPORTANT: JSON keys must be 0-indexed class IDs (i.e., pixel_id - 1).
        For example: pixel_id 123 (water_tank) -> class_id 122 in the JSON.
        
        Classes not specified in the JSON default to weight 1.0.
        
        Returns:
            tuple: (weights tensor, list of rooftop-only class IDs)
        """
        if weights_path is None:
            return None, []
        
        weights_file = Path(weights_path)
        if not weights_file.exists():
            print(f"Warning: Class weights file not found at {weights_path}. Using uniform weights.")
            return None, []
        
        try:
            with open(weights_file, 'r') as f:
                data = json.load(f)
            
            # Use median frequency weights (recommended for balanced training)
            median_weights = data.get('median_frequency_weights', {})
            
            # Load rooftop-only class IDs (classes not annotated in ADE20K original)
            rooftop_class_ids = data.get('rooftop_only_class_ids', [])
            
            if not median_weights:
                print("Warning: 'median_frequency_weights' not found in JSON. Using uniform weights.")
                return None, rooftop_class_ids
            
            # Create weight tensor - default all classes to 1.0
            weights = torch.ones(num_labels)
            loaded_classes = []
            
            for class_id_str, weight in median_weights.items():
                # Skip comment/metadata keys
                if class_id_str.startswith('_'):
                    continue
                try:
                    class_id = int(class_id_str)
                    if 0 <= class_id < num_labels:
                        weights[class_id] = float(weight)
                        loaded_classes.append((class_id, weight))
                except (ValueError, TypeError):
                    continue
            
            print(f"✓ Loaded class weights from {weights_path}")
            print(f"  Applied weights to {len(loaded_classes)} classes (others default to 1.0)")
            print(f"  Rooftop-only classes (masked for ADE20K): {len(rooftop_class_ids)}")
            
            # Show weighted classes for verification
            weighted_above_1 = [(c, w) for c, w in loaded_classes if w > 1.0]
            if weighted_above_1:
                print(f"  Classes with weight > 1.0: {len(weighted_above_1)}")
                for cid, w in sorted(weighted_above_1, key=lambda x: -x[1])[:5]:
                    print(f"    class {cid}: {w:.2f}")
            
            return weights, rooftop_class_ids
            
        except Exception as e:
            print(f"Error loading class weights from {weights_path}: {e}")
            return None, []

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
    ):
        """
        Normalize target masks to shape [num_instances, H, W] (no channel dim),
        ensure dtype/device match with masks_queries_logits, and then call the
        Mask2Former matcher. Add diagnostics on failure.
        """

        def _normalize_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
            # Acceptable input shapes (per-target):
            #   [num_instances, 1, H, W]  -> squeeze -> [num_instances, H, W]
            #   [num_instances, H, W]     -> keep as-is
            # Defensive: if mask is single-instance [1, H, W] or [H, W], handle too.
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)

            # Move channel dim if present at pos 1 and equals 1
            if mask.ndim == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
            # If someone produced [1, H, W] (single instance without batch dim)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)  # [1, H, W] -> [H, W]
                # bring back to [N, H, W] with N=1 for consistency
                mask = mask.unsqueeze(0)

            # Final sanity: we expect mask to be [num_instances, H, W]
            if mask.ndim != 3:
                raise ValueError(
                    f"Unexpected mask ndim {mask.ndim}; expected 3 (num_instances, H, W)."
                )
            return mask

        # normalize mask tensors and ensure dtype/device compatibility
        mask_labels = []
        for i, target in enumerate(targets):
            if "masks" not in target:
                raise KeyError(f"target[{i}] is missing 'masks' key")

            mask_t = _normalize_mask_tensor(target["masks"])

            # ensure dtype/device consistent with model logits (float for sampling computations)
            mask_t = mask_t.to(dtype=masks_queries_logits.dtype, device=masks_queries_logits.device)
            mask_labels.append(mask_t)

        # class labels (long on correct device)
        class_labels = [target["labels"].long().to(masks_queries_logits.device) for target in targets]
        
        # extract is_rooftop flags for sample-aware loss weighting
        is_rooftop_flags = [target.get("is_rooftop", True) for target in targets]

        # call matcher with diagnostics on failure
        try:
            indices = self.matcher(
                masks_queries_logits=masks_queries_logits,
                mask_labels=mask_labels,
                class_queries_logits=class_queries_logits,
                class_labels=class_labels,
            )
        except RuntimeError as e:
            # Print shapes that are most likely relevant to the grid_sample error
            print("\n🔥 Mask2Former Matcher failed.")
            print("  masks_queries_logits:", getattr(masks_queries_logits, "shape", None))
            if class_queries_logits is not None:
                print("  class_queries_logits:", getattr(class_queries_logits, "shape", None))
            for i, (m, t) in enumerate(zip(mask_labels, targets)):
                print(f"  normalized target[{i}] mask shape: {m.shape}, dtype: {m.dtype}, device: {m.device}")
                if "labels" in t:
                    print(f"    target[{i}] labels shape: {t['labels'].shape}, dtype: {t['labels'].dtype}")
            # re-raise so you still get the full original traceback
            raise

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels_weighted(class_queries_logits, class_labels, indices, is_rooftop_flags)

        return {**loss_masks, **loss_classes}



    def loss_labels_weighted(self, class_queries_logits, class_labels, indices, is_rooftop_flags):
        """
        Sample-aware cross-entropy loss.
        For ADE20K samples (is_rooftop=False):
          - If ade_asymmetric_loss=True: Only penalize false positives for rooftop-only classes
          - If ade_asymmetric_loss=False: Use ade20k_weight (rooftop classes weighted to 0)
        For rooftop samples (is_rooftop=True), use full empty_weight.
        """
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        
        # Build per-sample target tensor
        idx = self._get_predictions_permutation_indices(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(class_labels, indices)])
        target_classes = torch.full(
            (batch_size, num_queries), self.num_labels,
            dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        
        # Compute loss per sample with appropriate weights
        loss_ce = torch.tensor(0.0, device=pred_logits.device)
        for b in range(batch_size):
            if is_rooftop_flags[b]:
                # Rooftop sample: use full weights
                sample_loss = torch.nn.functional.cross_entropy(
                    pred_logits[b], target_classes[b], weight=self.empty_weight
                )
            elif self.ade_asymmetric_loss:
                # ADE20K sample with asymmetric loss: only penalize false positives for rooftop classes
                sample_loss = self._asymmetric_cross_entropy(
                    pred_logits[b], target_classes[b]
                )
            else:
                # ADE20K sample with standard weighted loss
                sample_loss = torch.nn.functional.cross_entropy(
                    pred_logits[b], target_classes[b], weight=self.ade20k_weight
                )
            loss_ce = loss_ce + sample_loss
        
        loss_ce = loss_ce / batch_size
        return {"loss_cross_entropy": loss_ce}

    def _asymmetric_cross_entropy(self, logits, targets):
        """
        Asymmetric cross-entropy for ADE20K samples.
        For rooftop-only classes:
          - Penalize false positives (predicting rooftop class when target is non-rooftop)
          - Do NOT penalize false negatives (missing rooftop class predictions)
        For other classes: use normal weighted cross-entropy.
        
        Args:
            logits: [num_queries, num_classes+1] prediction logits
            targets: [num_queries] target class indices
        """
        num_queries = logits.shape[0]
        num_classes = self.num_labels + 1
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Create rooftop class mask
        rooftop_mask = torch.zeros(num_classes, dtype=torch.bool, device=logits.device)
        for cid in self.rooftop_class_ids:
            if 0 <= cid < self.num_labels:
                rooftop_mask[cid] = True
        
        loss = torch.tensor(0.0, device=logits.device)
        
        for q in range(num_queries):
            target_class = targets[q].item()
            
            # Get class weight for this target
            weight = self.empty_weight[target_class]
            
            if target_class < self.num_labels and rooftop_mask[target_class]:
                # Target is a rooftop-only class (shouldn't happen in ADE20K, but handle it)
                # Don't penalize at all since it's unlabeled in ADE20K
                continue
            else:
                # Target is a non-rooftop class or no-object
                # Standard cross-entropy for this query
                query_loss = -log_probs[q, target_class] * weight
                
                # Additionally penalize if model predicts rooftop classes (false positive)
                # by adding extra loss for high probability on rooftop classes
                probs = torch.exp(log_probs[q])
                for cid in self.rooftop_class_ids:
                    if 0 <= cid < self.num_labels:
                        # Penalize false positive: predicting rooftop class when target is not
                        # Use KL-divergence style penalty: p * log(p) to penalize high confidence
                        fp_penalty = probs[cid] * log_probs[q, cid]
                        query_loss = query_loss - fp_penalty  # subtract because log_probs is negative
                
                loss = loss + query_loss
        
        return loss / num_queries

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total  # type: ignore
