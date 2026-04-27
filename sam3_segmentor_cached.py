import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmengine.structures import PixelData, InstanceData
from mmseg.registry import MODELS
from PIL import Image
from config import Config

from core.grounding_execution_engine import GroundingExecutionEngine
from core.prompt_bank import PromptBank


@MODELS.register_module()
class CachedSAM3OpenSegmentor(BaseSegmentor):
    def __init__(
        self,
        classname_path,
        device="cuda",
        prob_thd=0.0,
        bg_idx=0,
        slide_stride=0,
        slide_crop=0,
        confidence_threshold=0.5,
        use_sem_seg=True,
        use_presence_score=True,
        use_transformer_decoder=True,
        semantic_instance_fusion="max",
        upsnet_conflict_weight=4.0,
        upsnet_confidence_weight=2.0,
        upsnet_unknown_bias=0.0,
        expanded_prompt_pool_path=None,
        optimize_method="guided",
        cache_text_embeddings=True,
        enable_expanded_prompt=False,
        prompt_batch_size=0,
        image_batch_size=0,
        slot_batch_size=0,
        auto_tune_slot_batch=False,
        execution_mode="auto",
        group_images_by_size=True,
        shared_image_encoder_batch=False,
        compile_model=False,
        processor_resolution=1008,
        inference_dtype="fp32",
        image_encoder_dtype=None,
        class_aggregation="max",
        class_aggregation_topk=2,
        class_aggregation_temperature=0.4,
        class_aggregation_gem_power=6.0,
        class_aggregation_consensus_beta=8.0,
        class_aggregation_reliability_alpha=0.7,
        class_aggregation_coverage_penalty=0.15,
        class_aggregation_consensus_boost=0.25,
        class_aggregation_support_ratio=0.6,
        class_aggregation_boost_agreement_threshold=0.65,
        class_aggregation_boost_max_coverage=0.45,
        class_aggregation_keep_ratio=0.9,
        class_aggregation_max_selected=0,
        class_aggregation_second_boost_scale=0.2,
        class_aggregation_second_suppress_scale=0.0,
        class_aggregation_mean_class_ids=None,
        ccpea_prompt_fusion_alpha=0.6,
        ccpea_presence_weight=0.7,
        ccpea_instance_weight=0.3,
        ccpea_uncertainty_weight=0.5,
        ccpea_consensus_weight=0.3,
        ccpea_pool_size=8,
        ccpea_topk=2,
        router_enabled=False,
        router_coarse_resolution=512,
        router_inference_dtype="bf16",
        router_mode="class",
        router_keep_per_class=1,
        router_global_topk=4,
        router_refine_topk_classes=4,
        router_min_refine_classes=0,
        router_refine_score_threshold=0.0,
        mask_query_chunk_size=32,
        slot_chunk_size=4,
        max_cross_image_slots=0,
        max_mask_tensor_mb=1024,
        mask_decoder_queries=200,
        mask_decoder_resolution=288,
        **kwargs,
    ):
        super().__init__()

        self.device = device if isinstance(device, torch.device) else torch.device(device)

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(
            bpe_path="/data/public/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path="/data/public/sam3/sam3.pt",
            compile=compile_model,
        )
        model = model.to(self.device)
        self.processor = Sam3Processor(
            model,
            resolution=processor_resolution,
            confidence_threshold=confidence_threshold,
            device=self.device,
        )

        self.prompt_bank = PromptBank(
            processor=self.processor,
            classname_path=classname_path,
            device=self.device,
            expanded_prompt_pool_path=expanded_prompt_pool_path,
            cache_text_embeddings=cache_text_embeddings,
            enable_expanded_prompt=enable_expanded_prompt,
            inference_dtype=inference_dtype,
        )
        self.query_words = self.prompt_bank.query_words
        self.query_idx = self.prompt_bank.query_idx
        self.text_embeddings = self.prompt_bank.text_embeddings
        self.cache_key = self.prompt_bank.cache_key
        self.num_cls = self.prompt_bank.num_cls
        self.num_queries = self.prompt_bank.num_queries

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder
        self.semantic_instance_fusion = str(semantic_instance_fusion).strip().lower()
        
        self.upsnet_conflict_weight = float(upsnet_conflict_weight)
        self.upsnet_confidence_weight = float(upsnet_confidence_weight)
        self.upsnet_unknown_bias = float(upsnet_unknown_bias)
        if not self.use_sem_seg and not self.use_transformer_decoder:
            raise ValueError(
                "CachedSAM3OpenSegmentor requires at least one enabled output "
                "branch: use_sem_seg or use_transformer_decoder."
            )
        if self.semantic_instance_fusion not in {
            "max",
            "pgrf",
        }:
            raise ValueError(
                "semantic_instance_fusion must be one of 'max' or 'pgrf'."
            )
        self.cache_text_embeddings = cache_text_embeddings
        self.enable_expanded_prompt = enable_expanded_prompt
        self.optimize_method = optimize_method
        self.prompt_batch_size = (
            int(prompt_batch_size) if prompt_batch_size else self.num_queries
        )
        self.prompt_batch_size = max(1, min(self.prompt_batch_size, self.num_queries))
        self.image_batch_size = int(image_batch_size) if image_batch_size else 0
        self.slot_batch_size = int(slot_batch_size) if slot_batch_size else 0
        self.auto_tune_slot_batch = auto_tune_slot_batch
        self.execution_mode = execution_mode
        self.group_images_by_size = group_images_by_size
        self.shared_image_encoder_batch = shared_image_encoder_batch
        self.compile_model = compile_model
        self.processor_resolution = int(processor_resolution)
        self.inference_dtype = inference_dtype
        self.image_encoder_dtype = image_encoder_dtype
        self.class_aggregation = str(class_aggregation).strip().lower()
        if self.class_aggregation not in {"max", "topk_mean"}:
            raise ValueError(
                "class_aggregation must be one of 'max' or 'topk_mean'."
            )
        self.class_aggregation_topk = max(1, int(class_aggregation_topk))
        self.class_aggregation_temperature = float(class_aggregation_temperature)
        self.class_aggregation_gem_power = float(class_aggregation_gem_power)
        self.class_aggregation_consensus_beta = float(class_aggregation_consensus_beta)
        self.class_aggregation_reliability_alpha = float(class_aggregation_reliability_alpha)
        self.class_aggregation_coverage_penalty = float(
            class_aggregation_coverage_penalty
        )
        self.class_aggregation_consensus_boost = float(
            class_aggregation_consensus_boost
        )
        self.class_aggregation_support_ratio = float(class_aggregation_support_ratio)
        self.class_aggregation_boost_agreement_threshold = float(
            class_aggregation_boost_agreement_threshold
        )
        self.class_aggregation_boost_max_coverage = float(
            class_aggregation_boost_max_coverage
        )
        self.class_aggregation_keep_ratio = float(class_aggregation_keep_ratio)
        self.class_aggregation_max_selected = (
            int(class_aggregation_max_selected) if class_aggregation_max_selected else 0
        )
        self.class_aggregation_second_boost_scale = float(
            class_aggregation_second_boost_scale
        )
        self.class_aggregation_second_suppress_scale = float(
            class_aggregation_second_suppress_scale
        )
        if class_aggregation_mean_class_ids is None:
            self.class_aggregation_mean_class_ids = set()
        else:
            self.class_aggregation_mean_class_ids = {
                int(cls_id) for cls_id in class_aggregation_mean_class_ids
            }
        self.ccpea_prompt_fusion_alpha = float(ccpea_prompt_fusion_alpha)
        self.ccpea_presence_weight = float(ccpea_presence_weight)
        self.ccpea_instance_weight = float(ccpea_instance_weight)
        self.ccpea_uncertainty_weight = float(ccpea_uncertainty_weight)
        self.ccpea_consensus_weight = float(ccpea_consensus_weight)
        self.ccpea_pool_size = max(1, int(ccpea_pool_size))
        self.ccpea_topk = max(1, int(ccpea_topk))
        self.router_enabled = router_enabled
        self.router_coarse_resolution = int(router_coarse_resolution)
        self.router_inference_dtype = router_inference_dtype
        self.router_mode = router_mode
        self.router_keep_per_class = max(0, int(router_keep_per_class))
        self.router_global_topk = max(0, int(router_global_topk))
        self.router_refine_topk_classes = max(0, int(router_refine_topk_classes))
        self.router_min_refine_classes = max(0, int(router_min_refine_classes))
        self.router_refine_score_threshold = float(router_refine_score_threshold)
        self.mask_query_chunk_size = max(1, int(mask_query_chunk_size))
        self.slot_chunk_size = max(1, int(slot_chunk_size))
        self.max_cross_image_slots = int(max_cross_image_slots) if max_cross_image_slots else 0
        self.max_mask_tensor_mb = float(max_mask_tensor_mb)
        self.mask_decoder_queries = int(mask_decoder_queries)
        self.mask_decoder_resolution = int(mask_decoder_resolution)
        self._engine_use_presence_score = False

        self.execution_engine = GroundingExecutionEngine(
            processor=self.processor,
            prompt_bank=self.prompt_bank,
            device=self.device,
            prompt_batch_size=self.prompt_batch_size,
            image_batch_size=self.image_batch_size,
            slot_batch_size=self.slot_batch_size,
            auto_tune_slot_batch=self.auto_tune_slot_batch,
            execution_mode=self.execution_mode,
            group_images_by_size=self.group_images_by_size,
            shared_image_encoder_batch=self.shared_image_encoder_batch,
            confidence_threshold=self.confidence_threshold,
            use_sem_seg=self.use_sem_seg,
            use_presence_score=self._engine_use_presence_score,
            use_transformer_decoder=self.use_transformer_decoder,
        
            inference_dtype=self.inference_dtype,
            image_encoder_dtype=self.image_encoder_dtype,
            mask_query_chunk_size=self.mask_query_chunk_size,
            slot_chunk_size=self.slot_chunk_size,
            max_cross_image_slots=self.max_cross_image_slots,
            max_mask_tensor_mb=self.max_mask_tensor_mb,
            mask_decoder_queries=self.mask_decoder_queries,
            mask_decoder_resolution=self.mask_decoder_resolution,
        )
        self.coarse_processor = None
        self.coarse_execution_engine = None
        if self.router_enabled:
            self.coarse_processor = Sam3Processor(
                model,
                resolution=self.router_coarse_resolution,
                confidence_threshold=confidence_threshold,
                device=self.device,
            )
            self.coarse_execution_engine = GroundingExecutionEngine(
                processor=self.coarse_processor,
                prompt_bank=self.prompt_bank,
                device=self.device,
                prompt_batch_size=self.prompt_batch_size,
                image_batch_size=1,
                slot_batch_size=0,
                auto_tune_slot_batch=False,
                execution_mode="per_image",
                group_images_by_size=self.group_images_by_size,
                shared_image_encoder_batch=False,
                confidence_threshold=self.confidence_threshold,
                use_sem_seg=self.use_sem_seg,
                use_presence_score=self._engine_use_presence_score,
                use_transformer_decoder=self.use_transformer_decoder,
                inference_dtype=self.router_inference_dtype,
                image_encoder_dtype=self.router_inference_dtype,
                mask_query_chunk_size=self.mask_query_chunk_size,
                slot_chunk_size=self.slot_chunk_size,
                max_cross_image_slots=0,
                max_mask_tensor_mb=self.max_mask_tensor_mb,
                mask_decoder_queries=self.mask_decoder_queries,
                mask_decoder_resolution=self.mask_decoder_resolution,
            )

    def _inference_single_view(self, image):
        component_outputs = self.execution_engine.infer_images(
            [image], return_components=True
        )[0]
        return self._fuse_prompt_outputs(component_outputs)

    def _inference_multi_view(self, images):
        component_outputs = self.execution_engine.infer_images(
            images, return_components=True
        )
        return [self._fuse_prompt_outputs(outputs) for outputs in component_outputs]

    def _route_query_indices(self, coarse_logits):
        prompt_scores = coarse_logits.flatten(1).amax(dim=1)
        selected = set()

        if self.router_keep_per_class > 0:
            for cls_idx in range(self.num_cls):
                class_mask = self.query_idx == cls_idx
                class_query_indices = torch.nonzero(class_mask, as_tuple=False).flatten()
                if class_query_indices.numel() == 0:
                    continue
                class_scores = prompt_scores.index_select(0, class_query_indices)
                topk = min(self.router_keep_per_class, class_query_indices.numel())
                top_indices = class_scores.topk(topk).indices
                selected.update(class_query_indices.index_select(0, top_indices).tolist())

        if self.router_global_topk > 0:
            global_topk = min(self.router_global_topk, prompt_scores.numel())
            selected.update(prompt_scores.topk(global_topk).indices.tolist())

        if not selected:
            selected = set(range(self.num_queries))

        selected_query_indices = torch.tensor(
            sorted(selected), device=self.device, dtype=torch.long
        )
        return selected_query_indices

    def _aggregate_consensus_gem_probs(
        self,
        selected_probs,
        selected_rel,
        cls_idx=None,
        selected_coverage=None,
    ):
        eps = 1e-6
        max_probs = selected_probs.amax(dim=0)
        if selected_probs.shape[0] == 1:
            return max_probs

        weight_logits = (
            self.class_aggregation_consensus_beta * selected_probs
            + selected_rel.clamp_min(eps).log().view(-1, 1, 1)
        )
        weights = torch.softmax(weight_logits, dim=0)
        gem_power = max(self.class_aggregation_gem_power, 1.0)
        gem_probs = (
            (weights * selected_probs.pow(gem_power)).sum(dim=0).clamp_min(eps)
        ).pow(1.0 / gem_power)

        weighted_mean = (weights * selected_probs).sum(dim=0)
        weighted_abs_dev = (
            weights * (selected_probs - weighted_mean.unsqueeze(0)).abs()
        ).sum(dim=0)
        agreement = (
            1.0 - weighted_abs_dev / max_probs.clamp_min(eps)
        ).clamp(0.0, 1.0)
        support = (
            selected_probs
            >= self.class_aggregation_support_ratio * max_probs.unsqueeze(0)
        ).float().mean(dim=0).clamp(0.0, 1.0)

        # When one prompt clearly dominates globally, stay close to max.
        leader_margin = (
            (selected_rel[0] - selected_rel[1]).clamp_min(0.0)
            / selected_rel[0].clamp_min(eps)
        )
        global_balance = (1.0 - leader_margin).clamp(0.0, 1.0)

        # Disagreement triggers conservative shrinkage toward GeM.
        conservative_blend = (agreement * (1.0 - support)).clamp(0.0, 1.0)
        conservative_probs = torch.lerp(max_probs, gem_probs, conservative_blend)

        global_agreement = float((agreement * support).mean().item())
        mean_coverage = (
            float(selected_coverage.mean().item())
            if selected_coverage is not None
            else 0.0
        )
        boost_gate = 0.0
        if (
            global_agreement >= self.class_aggregation_boost_agreement_threshold
            and mean_coverage <= self.class_aggregation_boost_max_coverage
        ):
            agreement_range = max(
                1.0 - self.class_aggregation_boost_agreement_threshold, eps
            )
            boost_gate = min(
                1.0,
                (
                    global_agreement - self.class_aggregation_boost_agreement_threshold
                )
                / agreement_range,
            )

        # Strong local consensus is allowed to mildly exceed max only when
        # the whole class response is compact and globally self-consistent.
        consensus_gate = (
            global_balance * agreement * support.pow(2) * boost_gate
        ).clamp(0.0, 1.0)
        consensus_gain = (
            self.class_aggregation_consensus_boost
            * weighted_mean
            * (1.0 - max_probs)
            * consensus_gate
        )
        boosted_probs = (max_probs + consensus_gain).clamp(max=1.0)

        final_probs = torch.maximum(conservative_probs, boosted_probs)
        if cls_idx is not None and int(cls_idx) == int(self.bg_idx):
            # Keep background conservative; aggressive boosting should focus on foreground.
            final_probs = torch.maximum(final_probs, max_probs)
        return final_probs

    def _aggregate_max_second_agreement_probs(
        self,
        selected_probs,
        cls_idx=None,
        selected_coverage=None,
    ):
        eps = 1e-6
        max_probs = selected_probs.amax(dim=0)
        if selected_probs.shape[0] == 1:
            return max_probs

        top2 = selected_probs.topk(min(2, selected_probs.shape[0]), dim=0).values
        second_probs = top2[1]
        agreement = (second_probs / max_probs.clamp_min(eps)).clamp(0.0, 1.0)

        boost_gate = 1.0
        if selected_coverage is not None:
            mean_coverage = float(selected_coverage.mean().item())
            if mean_coverage > self.class_aggregation_boost_max_coverage:
                boost_gate = 0.0

        boost = (
            self.class_aggregation_second_boost_scale
            * boost_gate
            * agreement.pow(2)
            * second_probs
            * (1.0 - max_probs)
        )

        suppress = 0.0
        if self.class_aggregation_second_suppress_scale > 0:
            gap = (max_probs - second_probs).clamp_min(0.0)
            suppress = (
                self.class_aggregation_second_suppress_scale
                * (1.0 - agreement).pow(2)
                * gap
            )

        final_probs = (max_probs - suppress + boost).clamp(0.0, 1.0)
        if cls_idx is not None and int(cls_idx) == int(self.bg_idx):
            return max_probs
        return final_probs

    def _aggregate_max_consensus_gem_probs(
        self,
        selected_probs,
        selected_rel,
        cls_idx=None,
        selected_coverage=None,
    ):
        eps = 1e-6
        max_probs = selected_probs.amax(dim=0)
        if selected_probs.shape[0] == 1:
            return max_probs

        weight_logits = (
            self.class_aggregation_consensus_beta * selected_probs
            + selected_rel.clamp_min(eps).log().view(-1, 1, 1)
        )
        weights = torch.softmax(weight_logits, dim=0)
        gem_power = max(self.class_aggregation_gem_power, 1.0)
        gem_probs = (
            (weights * selected_probs.pow(gem_power)).sum(dim=0).clamp_min(eps)
        ).pow(1.0 / gem_power)

        top2 = selected_probs.topk(min(2, selected_probs.shape[0]), dim=0).values
        second_probs = top2[1]

        second_agreement = (second_probs / max_probs.clamp_min(eps)).clamp(0.0, 1.0)
        gem_agreement = (gem_probs / max_probs.clamp_min(eps)).clamp(0.0, 1.0)
        agreement = torch.minimum(second_agreement, gem_agreement)

        support_base = torch.minimum(second_probs, gem_probs)

        boost_gate = torch.ones_like(max_probs)
        if selected_coverage is not None:
            mean_coverage = float(selected_coverage.mean().item())
            if mean_coverage > self.class_aggregation_boost_max_coverage:
                boost_gate = torch.zeros_like(max_probs)

        pixel_gate = (
            agreement >= self.class_aggregation_boost_agreement_threshold
        ).float()
        boost = (
            self.class_aggregation_consensus_boost
            * boost_gate
            * pixel_gate
            * agreement.pow(2)
            * support_base
            * (1.0 - max_probs)
        )

        final_probs = (max_probs + boost).clamp(0.0, 1.0)
        if cls_idx is not None and int(cls_idx) == int(self.bg_idx):
            return max_probs
        return final_probs

    def _aggregate_noisy_or_probs(self, selected_probs, selected_weights=None):
        eps = 1e-6
        selected_probs = selected_probs.clamp(eps, 1.0 - eps)
        if selected_probs.shape[0] == 1:
            return selected_probs[0]

        if selected_weights is None:
            selected_weights = torch.full(
                (selected_probs.shape[0],),
                1.0 / selected_probs.shape[0],
                device=selected_probs.device,
                dtype=selected_probs.dtype,
            )
        else:
            selected_weights = selected_weights.to(
                device=selected_probs.device,
                dtype=selected_probs.dtype,
            )
            selected_weights = (
                selected_weights / selected_weights.sum().clamp_min(eps)
            )

        weighted_log_complement = (
            selected_weights.view(-1, 1, 1) * torch.log1p(-selected_probs)
        ).sum(dim=0)
        return (-torch.expm1(weighted_log_complement)).clamp(eps, 1.0)

    def _aggregate_support_boost_probs(self, selected_probs, selected_rel):
        eps = 1e-6
        max_probs = selected_probs.amax(dim=0)
        if selected_probs.shape[0] == 1:
            return max_probs

        rel_weights = torch.softmax(selected_rel / 0.1, dim=0)
        weighted_mean = (
            rel_weights.view(-1, 1, 1) * selected_probs
        ).sum(dim=0)
        weighted_abs_dev = (
            rel_weights.view(-1, 1, 1)
            * (selected_probs - weighted_mean.unsqueeze(0)).abs()
        ).sum(dim=0)
        agreement = (
            1.0 - weighted_abs_dev / max_probs.clamp_min(eps)
        ).clamp(0.0, 1.0)
        support = (
            selected_probs
            >= self.class_aggregation_support_ratio * max_probs.unsqueeze(0)
        ).float().mean(dim=0).clamp(0.0, 1.0)
        boost = (
            0.25
            * agreement
            * support.pow(2)
            * weighted_mean
            * (1.0 - max_probs)
        )
        return (max_probs + boost).clamp(eps, 1.0)

    def _select_reliable_prompt_indices(self, reliability_scores):
        if reliability_scores.numel() == 0:
            return reliability_scores.new_empty((0,), dtype=torch.long)

        sorted_scores, sorted_indices = reliability_scores.sort(descending=True)
        base_keep = min(self.class_aggregation_topk, reliability_scores.numel())
        best_score = sorted_scores[0].clamp_min(1e-6)
        keep_mask = sorted_scores >= best_score * self.class_aggregation_keep_ratio
        keep_count = max(base_keep, int(keep_mask.sum().item()))
        if self.class_aggregation_max_selected > 0:
            keep_count = min(keep_count, self.class_aggregation_max_selected)
        return sorted_indices[:keep_count]

    def _aggregate_query_logits_to_class_logits(self, query_logits, query_indices=None):
        if query_indices is None:
            query_indices = self.query_idx
        if not isinstance(query_indices, torch.Tensor):
            query_indices = torch.tensor(
                query_indices, device=self.device, dtype=torch.long
            )
        else:
            query_indices = query_indices.to(device=self.device, dtype=torch.long)

        if query_logits.shape[0] == self.num_cls and query_indices.numel() == self.num_cls:
            return query_logits

        class_logits = torch.full(
            (self.num_cls, *query_logits.shape[-2:]),
            float("-inf"),
            device=query_logits.device,
            dtype=query_logits.dtype,
        )
        query_scores = query_logits.flatten(1).amax(dim=1)
        for cls_idx in range(self.num_cls):
            cls_mask = query_indices == cls_idx
            if cls_mask.any():
                cls_query_logits = query_logits[cls_mask]
                if self.class_aggregation == "max":
                    class_logits[cls_idx] = cls_query_logits.amax(dim=0)
                else:
                    cls_query_scores = query_scores[cls_mask]
                    topk = min(self.class_aggregation_topk, cls_query_logits.shape[0])
                    top_indices = cls_query_scores.topk(topk).indices
                    top_logits = cls_query_logits.index_select(0, top_indices)
                    class_logits[cls_idx] = top_logits.mean(dim=0)
        return class_logits

    def _fuse_prompt_outputs(self, component_outputs):
        if self.semantic_instance_fusion == "max":
            fused_probs = component_outputs["fused_logits"]
            if self.use_presence_score:
                fused_probs = (
                    fused_probs
                    * component_outputs["presence_scores"].view(-1, 1, 1)
                )
            return fused_probs
        if self.semantic_instance_fusion == "pgrf":
            return self._pgrf_prompt_fusion(component_outputs)
        raise ValueError(
            f"Unsupported semantic_instance_fusion: {self.semantic_instance_fusion}"
        )

    def _pgrf_prompt_fusion(self, component_outputs, eps=1e-4):
        semantic_probs = component_outputs["semantic_logits"]
        max_fused_probs = component_outputs["fused_logits"]
        presence_scores = component_outputs["presence_scores"]
        max_instance_scores = component_outputs["max_instance_scores"]

        agreement = 1.0 - (semantic_probs - max_fused_probs).abs()
        residual_gate = (
            presence_scores.view(-1, 1, 1)
            * max_instance_scores.view(-1, 1, 1)
            * agreement.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        instance_advantage = (max_fused_probs - semantic_probs).clamp(min=0.0)
        fused_probs = (semantic_probs + residual_gate * instance_advantage).clamp(
            eps, 1.0 - eps
        )
        if self.use_presence_score:
            fused_probs = fused_probs * presence_scores.view(-1, 1, 1)
        return fused_probs

    def _build_prediction_outputs(self, seg_logits):
        seg_pred = torch.argmax(seg_logits, dim=0)
        max_vals = seg_logits.max(0)[0]
        seg_pred[max_vals < self.prob_thd] = self.bg_idx

        masks_list = []
        scores_list = []
        labels_list = []
        for cls_idx in range(self.num_cls):
            if cls_idx == self.bg_idx:
                continue

            cls_logits = seg_logits[cls_idx]
            mask = (seg_pred == cls_idx) & (cls_logits >= self.prob_thd)
            if not mask.any():
                continue

            masks_list.append(mask.float())
            scores_list.append(cls_logits[mask].amax())
            labels_list.append(
                torch.tensor(cls_idx, device=seg_logits.device, dtype=torch.int64)
            )

        instances = InstanceData()
        if masks_list:
            instances.masks = torch.stack(masks_list, dim=0).cpu().numpy()
            instances.scores = torch.stack(scores_list, dim=0).cpu()
            instances.labels = torch.stack(labels_list, dim=0).cpu()
        else:
            height, width = seg_pred.shape[-2:]
            instances.masks = torch.zeros(
                (0, height, width), dtype=torch.float32
            ).cpu().numpy()
            instances.scores = torch.zeros((0,), dtype=torch.float32).cpu()
            instances.labels = torch.zeros((0,), dtype=torch.int64).cpu()

        return seg_pred, instances

    def _route_class_indices(self, coarse_class_logits):
        class_scores = coarse_class_logits.flatten(1).amax(dim=1)
        if self.router_refine_topk_classes <= 0 and self.router_min_refine_classes <= 0:
            return torch.arange(self.num_cls, device=self.device, dtype=torch.long)

        selected = set()
        if self.router_refine_score_threshold > 0:
            threshold_indices = torch.nonzero(
                class_scores >= self.router_refine_score_threshold, as_tuple=False
            ).flatten()
            selected.update(threshold_indices.tolist())

        if self.router_min_refine_classes > 0:
            min_topk = min(self.router_min_refine_classes, self.num_cls)
            selected.update(class_scores.topk(min_topk).indices.tolist())

        if self.router_refine_topk_classes > 0:
            max_topk = min(self.router_refine_topk_classes, self.num_cls)
            top_indices = class_scores.topk(max_topk).indices
            if selected:
                selected = {
                    cls_idx for cls_idx in selected if cls_idx in set(top_indices.tolist())
                }
            else:
                selected = set(top_indices.tolist())

        if not selected:
            selected = {int(class_scores.argmax().item())}

        return torch.tensor(sorted(selected), device=self.device, dtype=torch.long)

    def _get_query_indices_for_classes(self, class_indices):
        selected = []
        for cls_idx in class_indices.tolist():
            cls_query_indices = torch.nonzero(
                self.query_idx == cls_idx, as_tuple=False
            ).flatten()
            selected.extend(cls_query_indices.tolist())
        if not selected:
            selected = list(range(self.num_queries))
        return torch.tensor(sorted(set(selected)), device=self.device, dtype=torch.long)

    def _inference_single_view_routed(self, image):
        if not self.router_enabled or self.coarse_execution_engine is None:
            return self._inference_single_view(image)

        coarse_logits = self.coarse_execution_engine.infer_images([image])[0]
        if self.router_mode == "class":
            coarse_class_logits = self._aggregate_query_logits_to_class_logits(coarse_logits)
            selected_class_indices = self._route_class_indices(coarse_class_logits)
            selected_query_indices = self._get_query_indices_for_classes(selected_class_indices)
            fine_query_logits = self.execution_engine.infer_images(
                [image], query_indices=selected_query_indices
            )[0]
            fine_class_logits = self._aggregate_query_logits_to_class_logits(
                fine_query_logits, query_indices=self.query_idx.index_select(0, selected_query_indices)
            )
            final_class_logits = coarse_class_logits.clone()
            fused_selected_logits = torch.maximum(
                final_class_logits.index_select(0, selected_class_indices),
                fine_class_logits.index_select(0, selected_class_indices),
            )
            final_class_logits.index_copy_(
                0,
                selected_class_indices,
                fused_selected_logits,
            )
            return final_class_logits

        selected_query_indices = self._route_query_indices(coarse_logits)
        fine_logits = self.execution_engine.infer_images(
            [image], query_indices=selected_query_indices
        )[0]

        final_logits = coarse_logits.clone()
        final_logits.index_copy_(
            0,
            selected_query_indices,
            torch.maximum(
                final_logits.index_select(0, selected_query_indices),
                fine_logits,
            ),
        )
        return final_logits

    def slide_inference(self, image, stride, crop_size):
        w_img, h_img = image.size

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size

        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)

                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = image.crop((x1, y1, x2, y2))
                crop_seg_logit = self._inference_single_view(crop_img)

                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."
        return preds / count_mat

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                )
            ] * inputs.shape[0]

        images = []
        for meta in batch_img_metas:
            image_path = meta.get("img_path")
            images.append(Image.open(image_path).convert("RGB"))

        seg_logits_per_image = [None] * len(images)
        regular_indices = []
        regular_images = []
        for i, (image, meta) in enumerate(zip(images, batch_img_metas)):
            needs_sliding = self.slide_crop > 0 and (
                self.slide_crop < image.size[0] or self.slide_crop < image.size[1]
            )
            if needs_sliding:
                seg_logits_per_image[i] = self.slide_inference(
                    image, self.slide_stride, self.slide_crop
                )
            else:
                if self.router_enabled:
                    seg_logits_per_image[i] = self._inference_single_view_routed(image)
                else:
                    regular_indices.append(i)
                    regular_images.append(image)

        if regular_images:
            regular_logits = self._inference_multi_view(regular_images)
            for idx, seg_logits in zip(regular_indices, regular_logits):
                if (
                    self.class_aggregation != "max"
                    and seg_logits.shape[0] == self.num_queries
                ):
                    seg_logits = self._aggregate_query_logits_to_class_logits(
                        seg_logits
                    )
                seg_logits_per_image[idx] = seg_logits

        cls_index = None
        if self.num_cls != self.num_queries:
            cls_index = nn.functional.one_hot(
                self.query_idx, num_classes=self.num_cls
            ).T.view(self.num_cls, len(self.query_idx), 1, 1)

        for i, meta in enumerate(batch_img_metas):
            seg_logits = seg_logits_per_image[i]
            ori_shape = meta["ori_shape"]
            if isinstance(ori_shape, (list, tuple)) and len(ori_shape) > 2:
                ori_shape = ori_shape[:2]

            if seg_logits.shape[-2:] != tuple(ori_shape):
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0),
                    size=ori_shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            if (
                cls_index is not None
                and self.class_aggregation == "max"
                and seg_logits.shape[0] == self.num_queries
            ):
                seg_logits = (seg_logits.unsqueeze(0) * cls_index).max(1)[0]

            seg_pred, pred_instances = self._build_prediction_outputs(seg_logits)

            data_samples[i].set_data(
                {
                    "seg_logits": PixelData(**{"data": seg_logits}),
                    "pred_sem_seg": PixelData(**{"data": seg_pred.unsqueeze(0)}),
                }
            )
            data_samples[i].pred_instances = pred_instances

        return data_samples

    def _forward(self, data_samples):
        pass

    def inference(self, img, batch_img_metas):
        pass

    def encode_decode(self, inputs, batch_img_metas):
        pass

    def extract_feat(self, inputs):
        pass

    def loss(self, inputs, data_samples):
        pass

    def get_cls_idx(self, path):
        return PromptBank.get_cls_idx(path)


def get_cls_idx(path):
    return PromptBank.get_cls_idx(path)
