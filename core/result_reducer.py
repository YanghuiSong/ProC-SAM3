import torch
import torch.nn.functional as F


class GroundingResultReducer:
    """Fuses raw SAM3 outputs into per-slot segmentation logits."""

    def __init__(
        self,
        device,
        confidence_threshold=0.5,
        use_sem_seg=True,
        use_presence_score=True,
        use_transformer_decoder=True,
        semantic_instance_fusion="max",
        semantic_fusion_alpha=0.7,
        upsnet_conflict_weight=4.0,
        upsnet_confidence_weight=2.0,
        upsnet_unknown_bias=0.0,
        mask_query_chunk_size=32,
        slot_chunk_size=4,
    ):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder
        self.semantic_instance_fusion = str(semantic_instance_fusion).strip().lower()
        self.semantic_fusion_alpha = float(semantic_fusion_alpha)
        self.upsnet_conflict_weight = float(upsnet_conflict_weight)
        self.upsnet_confidence_weight = float(upsnet_confidence_weight)
        self.upsnet_unknown_bias = float(upsnet_unknown_bias)
        self.mask_query_chunk_size = max(1, int(mask_query_chunk_size))
        self.slot_chunk_size = max(1, int(slot_chunk_size))

    def _pgrf_fuse(
        self,
        semantic_component,
        instance_component,
        presence_scores,
        max_instance_scores,
    ):
        agreement = 1.0 - (semantic_component - instance_component).abs()
        residual_gate = (
            presence_scores.view(-1, 1, 1)
            * max_instance_scores.view(-1, 1, 1)
            * agreement.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        instance_advantage = (instance_component - semantic_component).clamp(min=0.0)
        return (semantic_component + residual_gate * instance_advantage).clamp(0.0, 1.0)

    def reduce_slot_outputs(self, outputs, slot_count, output_size, return_components=False):
        height, width = output_size
        slot_logits = torch.zeros((slot_count, height, width), device=self.device)
        semantic_component = torch.zeros((slot_count, height, width), device=self.device)
        instance_component = torch.zeros((slot_count, height, width), device=self.device)
        instance_best_component = torch.zeros((slot_count, height, width), device=self.device)
        max_instance_scores = torch.zeros(slot_count, device=self.device)

        presence_logits = outputs.get("presence_logit_dec")
        if presence_logits is None:
            presence_scores = torch.ones(slot_count, device=self.device)
        else:
            presence_scores = presence_logits.sigmoid().reshape(slot_count, -1)[:, 0]

        if self.use_sem_seg and "semantic_seg" in outputs:
            semantic_logits = outputs["semantic_seg"]
            if semantic_logits.ndim == 3:
                semantic_logits = semantic_logits.unsqueeze(1)
            if semantic_logits.shape[-2:] != (height, width):
                semantic_logits = F.interpolate(
                    semantic_logits,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            semantic_component = semantic_logits[:, 0].sigmoid()

        if (
            self.use_transformer_decoder
            and "pred_masks" in outputs
            and "pred_logits" in outputs
            and outputs["pred_masks"].shape[1] > 0
        ):
            masks = outputs["pred_masks"]
            if masks.ndim == 5 and masks.shape[2] == 1:
                masks = masks.squeeze(2)
            if masks.ndim != 4:
                raise RuntimeError(f"Unexpected pred_masks shape: {tuple(masks.shape)}")

            object_scores = outputs["pred_logits"].sigmoid()
            if object_scores.ndim == 3 and object_scores.shape[-1] == 1:
                object_scores = object_scores.squeeze(-1)
            object_scores = object_scores * presence_scores.view(-1, 1)
            valid_scores = object_scores > self.confidence_threshold
            max_instance_scores = object_scores.max(dim=1).values

            num_masks = masks.shape[1]
            for slot_start in range(0, slot_count, self.slot_chunk_size):
                slot_end = min(slot_start + self.slot_chunk_size, slot_count)
                chunk_masks = masks[slot_start:slot_end]
                chunk_scores = object_scores[slot_start:slot_end]
                chunk_valid = valid_scores[slot_start:slot_end]
                chunk_best = torch.zeros(
                    (slot_end - slot_start, height, width), device=self.device
                )
                chunk_noisy_or = torch.zeros(
                    (slot_end - slot_start, height, width), device=self.device
                )

                for query_start in range(0, num_masks, self.mask_query_chunk_size):
                    query_end = min(query_start + self.mask_query_chunk_size, num_masks)
                    valid_chunk = chunk_valid[:, query_start:query_end]
                    if not valid_chunk.any():
                        continue

                    sub_masks = chunk_masks[:, query_start:query_end]
                    if sub_masks.shape[-2:] != (height, width):
                        sub_masks = F.interpolate(
                            sub_masks.reshape(-1, 1, *sub_masks.shape[-2:]),
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        ).reshape(
                            slot_end - slot_start,
                            query_end - query_start,
                            height,
                            width,
                        )

                    sub_masks = sub_masks.sigmoid()
                    sub_scores = chunk_scores[:, query_start:query_end] * valid_chunk
                    sub_best = (
                        sub_masks * sub_scores.unsqueeze(-1).unsqueeze(-1)
                    ).amax(dim=1)
                    sub_weighted = sub_masks * sub_scores.unsqueeze(-1).unsqueeze(-1)
                    sub_noisy_or = 1.0 - torch.prod(
                        1.0 - sub_weighted.clamp(min=0.0, max=1.0), dim=1
                    )
                    chunk_best = torch.maximum(chunk_best, sub_best)
                    chunk_noisy_or = 1.0 - (
                        (1.0 - chunk_noisy_or) * (1.0 - sub_noisy_or)
                    )

                instance_component[slot_start:slot_end] = chunk_noisy_or
                instance_best_component[slot_start:slot_end] = chunk_best

        if self.use_sem_seg and self.use_transformer_decoder:
            if self.semantic_instance_fusion == "max":
                slot_logits = torch.maximum(
                    semantic_component,
                    instance_best_component,
                )
            elif self.semantic_instance_fusion == "pgrf":
                slot_logits = self._pgrf_fuse(
                    semantic_component,
                    instance_best_component,
                    presence_scores.clamp(0.0, 1.0),
                    max_instance_scores.clamp(0.0, 1.0),
                )
            else:
                raise ValueError(
                    f"Unsupported semantic_instance_fusion: {self.semantic_instance_fusion}"
                )
        elif self.use_sem_seg:
            slot_logits = semantic_component
        elif self.use_transformer_decoder:
            slot_logits = instance_best_component

        if self.use_presence_score:
            slot_logits *= presence_scores.view(-1, 1, 1)
            semantic_component *= presence_scores.view(-1, 1, 1)
            instance_component *= presence_scores.view(-1, 1, 1)

        if return_components:
            return {
                "fused_logits": slot_logits,
                "semantic_logits": semantic_component,
                "instance_logits": instance_component,
                "presence_scores": presence_scores,
                "max_instance_scores": max_instance_scores,
            }

        return slot_logits
