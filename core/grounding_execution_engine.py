from collections import OrderedDict
from contextlib import nullcontext

import torch

from sam3.model.data_misc import FindStage

from core.result_reducer import GroundingResultReducer


class GroundingExecutionEngine:
    """
    Multi-image x multi-prompt execution engine with adaptive slot batching.

    The engine packs image-prompt pairs into slot batches so the batch dimension
    inside SAM3 represents independent grounding slots instead of only prompts.
    """

    def __init__(
        self,
        processor,
        prompt_bank,
        device,
        prompt_batch_size=0,
        image_batch_size=0,
        slot_batch_size=0,
        auto_tune_slot_batch=False,
        execution_mode="auto",
        group_images_by_size=True,
        shared_image_encoder_batch=False,
        confidence_threshold=0.5,
        use_sem_seg=True,
        use_presence_score=True,
        use_transformer_decoder=True,
        inference_dtype="fp32",
        image_encoder_dtype=None,
        mask_query_chunk_size=32,
        slot_chunk_size=4,
        max_cross_image_slots=0,
        max_mask_tensor_mb=1024,
        mask_decoder_queries=200,
        mask_decoder_resolution=288,
    ):
        self.processor = processor
        self.model = processor.model
        self.prompt_bank = prompt_bank
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.num_queries = prompt_bank.num_queries
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
        self.use_transformer_decoder = use_transformer_decoder
        self.inference_dtype = inference_dtype
        self.image_encoder_dtype = image_encoder_dtype or inference_dtype
        self.max_cross_image_slots = int(max_cross_image_slots) if max_cross_image_slots else 0
        self.max_mask_tensor_bytes = int(float(max_mask_tensor_mb) * 1024 * 1024)
        self.mask_decoder_queries = int(mask_decoder_queries)
        self.mask_decoder_resolution = int(mask_decoder_resolution)
        self.reducer = GroundingResultReducer(
            device=self.device,
            confidence_threshold=confidence_threshold,
            use_sem_seg=use_sem_seg,
            use_presence_score=use_presence_score,
            use_transformer_decoder=use_transformer_decoder,
            mask_query_chunk_size=mask_query_chunk_size,
            slot_chunk_size=slot_chunk_size,
        )
        self._find_stage_cache = {}
        self._dummy_prompt_cache = {}
        self._effective_prompt_batch_cache = {}

        print(
            "GroundingExecutionEngine configured with "
            f"image_batch_size={self.image_batch_size or 'auto'}, "
            f"prompt_batch_size={self.prompt_batch_size}, "
            f"slot_batch_size={self.slot_batch_size or 'auto'}, "
            f"auto_tune_slot_batch={self.auto_tune_slot_batch}, "
            f"execution_mode={self.execution_mode}, "
            f"shared_image_encoder_batch={self.shared_image_encoder_batch}, "
            f"inference_dtype={self.inference_dtype}, "
            f"image_encoder_dtype={self.image_encoder_dtype}, "
            f"group_images_by_size={self.group_images_by_size}"
        )

    def infer_images(self, images, query_indices=None, return_components=False):
        if not images:
            return []

        selected_query_indices = self._normalize_query_indices(query_indices)
        results = [None] * len(images)
        for group_indices in self._group_indices(images):
            if self.image_batch_size > 0:
                image_groups = [
                    group_indices[i : i + self.image_batch_size]
                    for i in range(0, len(group_indices), self.image_batch_size)
                ]
            else:
                image_groups = [group_indices]

            for image_group in image_groups:
                group_images = [images[idx] for idx in image_group]
                if self._should_cross_image_batch(
                    group_images, total_prompt_count=selected_query_indices.numel()
                ):
                    group_logits = self._infer_same_size_group(
                        group_images,
                        query_indices=selected_query_indices,
                        return_components=return_components,
                    )
                elif self.shared_image_encoder_batch and len(group_images) > 1:
                    group_logits = self._infer_per_image_with_shared_encoder(
                        group_images,
                        query_indices=selected_query_indices,
                        return_components=return_components,
                    )
                else:
                    single_outputs = [
                        self._infer_single_image(
                            image,
                            query_indices=selected_query_indices,
                            return_components=return_components,
                        )
                        for image in group_images
                    ]
                    if return_components:
                        group_logits = self._stack_component_outputs(single_outputs)
                    else:
                        group_logits = torch.stack(single_outputs, dim=0)
                for local_idx, global_idx in enumerate(image_group):
                    if return_components:
                        results[global_idx] = {
                            key: value[local_idx] for key, value in group_logits.items()
                        }
                    else:
                        results[global_idx] = group_logits[local_idx]

        return results

    def _group_indices(self, images):
        if not self.group_images_by_size:
            size_set = {(image.height, image.width) for image in images}
            if len(size_set) == 1:
                return [list(range(len(images)))]
            return [[idx] for idx in range(len(images))]

        grouped = OrderedDict()
        for idx, image in enumerate(images):
            grouped.setdefault((image.height, image.width), []).append(idx)
        return list(grouped.values())

    def _infer_same_size_group(self, images, query_indices=None, return_components=False):
        if not images:
            return torch.empty(0, self.num_queries, 0, 0, device=self.device)

        expected_size = (images[0].height, images[0].width)
        if any((image.height, image.width) != expected_size for image in images):
            raise ValueError("GroundingExecutionEngine expects same-size images per group")

        with torch.no_grad():
            with self._encoder_autocast_context():
                inference_state = self.processor.set_image_batch(images)
            backbone_out_image_only = inference_state["backbone_out"]
            backbone_out_image_only = self._cast_backbone_out_to_decoder_dtype(
                backbone_out_image_only
            )
            self._prepare_cached_image_features(backbone_out_image_only)

            batch_size = len(images)
            height, width = expected_size
            seg_logits = self._run_prompt_batches(
                backbone_out_image_only=backbone_out_image_only,
                batch_size=batch_size,
                output_size=(height, width),
                query_indices=self._normalize_query_indices(query_indices),
                return_components=return_components,
            )

        return seg_logits

    def _infer_per_image_with_shared_encoder(
        self, images, query_indices=None, return_components=False
    ):
        if not images:
            return torch.empty(0, self.num_queries, 0, 0, device=self.device)

        expected_size = (images[0].height, images[0].width)
        if any((image.height, image.width) != expected_size for image in images):
            raise ValueError(
                "GroundingExecutionEngine expects same-size images per shared encoder group"
            )

        with torch.no_grad():
            with self._encoder_autocast_context():
                inference_state = self.processor.set_image_batch(images)
            backbone_out_image_only = inference_state["backbone_out"]
            backbone_out_image_only = self._cast_backbone_out_to_decoder_dtype(
                backbone_out_image_only
            )

            outputs = []
            normalized_query_indices = self._normalize_query_indices(query_indices)
            for image_index, image in enumerate(images):
                single_backbone_out = self._extract_single_image_backbone_out(
                    backbone_out_image_only, image_index
                )
                self._prepare_cached_image_features(single_backbone_out)
                outputs.append(
                    self._slice_batch_output(
                        self._run_prompt_batches(
                        backbone_out_image_only=single_backbone_out,
                        batch_size=1,
                        output_size=(image.height, image.width),
                        query_indices=normalized_query_indices,
                        return_components=return_components,
                        ),
                        batch_index=0,
                    )
                )

        if return_components:
            return self._stack_component_outputs(outputs)
        return torch.stack(outputs, dim=0)

    def _infer_single_image(self, image, query_indices=None, return_components=False):
        with torch.no_grad():
            with self._encoder_autocast_context():
                inference_state = self.processor.set_image(image)
            backbone_out_image_only = inference_state["backbone_out"]
            backbone_out_image_only = self._cast_backbone_out_to_decoder_dtype(
                backbone_out_image_only
            )
            self._prepare_cached_image_features(backbone_out_image_only)
            seg_logits = self._run_prompt_batches(
                backbone_out_image_only=backbone_out_image_only,
                batch_size=1,
                output_size=(image.height, image.width),
                query_indices=self._normalize_query_indices(query_indices),
                return_components=return_components,
            )
        return self._slice_batch_output(seg_logits, batch_index=0)

    def _run_prompt_batches(
        self, backbone_out_image_only, batch_size, output_size, query_indices, return_components
    ):
        height, width = output_size
        total_prompt_count = query_indices.numel()
        effective_prompt_batch = self._resolve_prompt_batch_size(
            num_images=batch_size,
            output_size=(height, width),
            backbone_out_image_only=backbone_out_image_only,
            total_prompt_count=total_prompt_count,
        )

        while True:
            try:
                if return_components:
                    seg_outputs = {
                        "fused_logits": torch.zeros(
                            (batch_size, total_prompt_count, height, width), device=self.device
                        ),
                        "semantic_logits": torch.zeros(
                            (batch_size, total_prompt_count, height, width), device=self.device
                        ),
                        "instance_logits": torch.zeros(
                            (batch_size, total_prompt_count, height, width), device=self.device
                        ),
                        "presence_scores": torch.zeros(
                            (batch_size, total_prompt_count), device=self.device
                        ),
                        "max_instance_scores": torch.zeros(
                            (batch_size, total_prompt_count), device=self.device
                        ),
                    }
                else:
                    seg_logits = torch.zeros(
                        (batch_size, total_prompt_count, height, width), device=self.device
                    )
                for local_start in range(0, total_prompt_count, effective_prompt_batch):
                    local_end = min(local_start + effective_prompt_batch, total_prompt_count)
                    chunk_query_indices = query_indices[local_start:local_end]
                    prompts_per_image = local_end - local_start
                    find_stage, dummy_prompt = self._get_prompt_batch_context(
                        batch_size, prompts_per_image
                    )
                    backbone_out = self.prompt_bank.build_backbone_out_by_indices(
                        backbone_out_image_only=backbone_out_image_only,
                        query_indices=chunk_query_indices,
                        image_repeats=batch_size,
                    )
                    with self._decoder_autocast_context():
                        outputs = self.model.forward_grounding(
                            backbone_out=backbone_out,
                            find_input=find_stage,
                            geometric_prompt=dummy_prompt,
                            find_target=None,
                        )
                    slot_logits = self.reducer.reduce_slot_outputs(
                        outputs=outputs,
                        slot_count=batch_size * prompts_per_image,
                        output_size=(height, width),
                        return_components=return_components,
                    )
                    if return_components:
                        reshaped_outputs = {
                            "fused_logits": slot_logits["fused_logits"].reshape(
                                batch_size, prompts_per_image, height, width
                            ),
                            "semantic_logits": slot_logits["semantic_logits"].reshape(
                                batch_size, prompts_per_image, height, width
                            ),
                            "instance_logits": slot_logits["instance_logits"].reshape(
                                batch_size, prompts_per_image, height, width
                            ),
                            "presence_scores": slot_logits["presence_scores"].reshape(
                                batch_size, prompts_per_image
                            ),
                            "max_instance_scores": slot_logits["max_instance_scores"].reshape(
                                batch_size, prompts_per_image
                            ),
                        }
                        for key, value in reshaped_outputs.items():
                            seg_outputs[key][:, local_start:local_end] = value
                    else:
                        seg_logits[:, local_start:local_end] = slot_logits.reshape(
                            batch_size, prompts_per_image, height, width
                        )
                break
            except torch.cuda.OutOfMemoryError:
                if effective_prompt_batch <= 1:
                    raise
                torch.cuda.empty_cache()
                effective_prompt_batch = max(1, effective_prompt_batch // 2)
                self._effective_prompt_batch_cache[
                    (batch_size, height, width, total_prompt_count)
                ] = (
                    effective_prompt_batch
                )
                print(
                    "OOM during execution, retrying with smaller effective prompt batch size "
                    f"{effective_prompt_batch} for {batch_size} images at {(height, width)}"
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or effective_prompt_batch <= 1:
                    raise
                torch.cuda.empty_cache()
                effective_prompt_batch = max(1, effective_prompt_batch // 2)
                self._effective_prompt_batch_cache[
                    (batch_size, height, width, total_prompt_count)
                ] = (
                    effective_prompt_batch
                )
                print(
                    "Runtime OOM during execution, retrying with smaller effective prompt batch size "
                    f"{effective_prompt_batch} for {batch_size} images at {(height, width)}"
                )

        if return_components:
            return seg_outputs
        return seg_logits

    def _should_cross_image_batch(self, images, total_prompt_count):
        num_images = len(images)
        if num_images <= 1:
            return True
        if self.execution_mode == "multi_image":
            return True
        if self.execution_mode == "per_image":
            return False

        estimated_slots = num_images * total_prompt_count
        if self.max_cross_image_slots > 0 and estimated_slots > self.max_cross_image_slots:
            return False
        if self.max_cross_image_slots <= 0 and estimated_slots > total_prompt_count:
            return False
        if not self.use_transformer_decoder:
            return True

        estimated_mask_bytes = (
            estimated_slots
            * self.mask_decoder_queries
            * self.mask_decoder_resolution
            * self.mask_decoder_resolution
            * 4
        )
        return estimated_mask_bytes <= self.max_mask_tensor_bytes

    def _resolve_prompt_batch_size(
        self, num_images, output_size, backbone_out_image_only, total_prompt_count
    ):
        cache_key = (num_images, *output_size, total_prompt_count)
        if not self.auto_tune_slot_batch:
            if cache_key in self._effective_prompt_batch_cache:
                return self._effective_prompt_batch_cache[cache_key]
            if self.slot_batch_size > 0:
                slot_limited_prompt_batch = max(
                    1, self.slot_batch_size // max(1, num_images)
                )
                effective_prompt_batch = min(
                    total_prompt_count, self.prompt_batch_size, slot_limited_prompt_batch
                )
            else:
                effective_prompt_batch = min(total_prompt_count, self.prompt_batch_size)
            self._effective_prompt_batch_cache[cache_key] = effective_prompt_batch
            print(
                "Resolved effective prompt batch size "
                f"{effective_prompt_batch} for {num_images} images at {output_size}"
            )
            return effective_prompt_batch

        if cache_key in self._effective_prompt_batch_cache:
            return self._effective_prompt_batch_cache[cache_key]

        if self.device.type != "cuda":
            resolved = min(total_prompt_count, self.prompt_batch_size)
            self._effective_prompt_batch_cache[cache_key] = resolved
            return resolved

        max_candidate = min(total_prompt_count, self.prompt_batch_size)
        for candidate in range(max_candidate, 0, -1):
            if self.slot_batch_size > 0 and candidate * num_images > self.slot_batch_size:
                continue
            try:
                self._probe_prompt_batch_size(
                    num_images=num_images,
                    prompts_per_image=candidate,
                    backbone_out_image_only=backbone_out_image_only,
                    output_size=output_size,
                )
                self._effective_prompt_batch_cache[cache_key] = candidate
                print(
                    "Auto-tuned effective prompt batch size "
                    f"{candidate} for {num_images} images at {output_size}"
                )
                return candidate
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                continue

        self._effective_prompt_batch_cache[cache_key] = 1
        print(
            "Auto-tuned effective prompt batch size fell back to 1 "
            f"for {num_images} images at {output_size}"
        )
        return 1

    def _probe_prompt_batch_size(
        self,
        num_images,
        prompts_per_image,
        backbone_out_image_only,
        output_size,
    ):
        find_stage, dummy_prompt = self._get_prompt_batch_context(
            num_images, prompts_per_image
        )
        backbone_out = self.prompt_bank.build_backbone_out(
            backbone_out_image_only=backbone_out_image_only,
            start_idx=0,
            end_idx=prompts_per_image,
            image_repeats=num_images,
        )
        with self._decoder_autocast_context():
            outputs = self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_stage,
                geometric_prompt=dummy_prompt,
                find_target=None,
            )
        _ = self.reducer.reduce_slot_outputs(
            outputs=outputs,
            slot_count=num_images * prompts_per_image,
            output_size=output_size,
        )

    def _get_prompt_batch_context(self, num_images, prompts_per_image):
        key = (num_images, prompts_per_image)
        if key not in self._find_stage_cache:
            slot_count = num_images * prompts_per_image
            self._find_stage_cache[key] = FindStage(
                img_ids=torch.arange(num_images, device=self.device, dtype=torch.long)
                .repeat_interleave(prompts_per_image),
                text_ids=torch.arange(slot_count, device=self.device, dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )

        slot_count = num_images * prompts_per_image
        if slot_count not in self._dummy_prompt_cache:
            self._dummy_prompt_cache[slot_count] = self.model._get_dummy_prompt(
                num_prompts=slot_count
            )

        return self._find_stage_cache[key], self._dummy_prompt_cache[slot_count]

    def _prepare_cached_image_features(self, backbone_out_image_only):
        num_levels = self.model.num_feature_levels
        vis_feats = backbone_out_image_only["backbone_fpn"][-num_levels:]
        vis_pos_enc = backbone_out_image_only["vision_pos_enc"][-num_levels:]
        backbone_out_image_only["_cached_img_feats"] = [
            feat.flatten(2).permute(2, 0, 1).contiguous() for feat in vis_feats
        ]
        backbone_out_image_only["_cached_img_pos_embeds"] = [
            pos.flatten(2).permute(2, 0, 1).contiguous() for pos in vis_pos_enc
        ]
        backbone_out_image_only["_cached_vis_feat_sizes"] = [
            pos.shape[-2:] for pos in vis_pos_enc
        ]

    def _extract_single_image_backbone_out(self, backbone_out_image_only, image_index):
        single_backbone_out = dict(backbone_out_image_only)
        single_backbone_out["backbone_fpn"] = [
            feat[image_index : image_index + 1] for feat in backbone_out_image_only["backbone_fpn"]
        ]
        single_backbone_out["vision_pos_enc"] = [
            pos[image_index : image_index + 1] for pos in backbone_out_image_only["vision_pos_enc"]
        ]

        if backbone_out_image_only.get("sam2_backbone_out") is not None:
            sam2_backbone_out = dict(backbone_out_image_only["sam2_backbone_out"])
            sam2_backbone_out["backbone_fpn"] = [
                feat[image_index : image_index + 1]
                for feat in backbone_out_image_only["sam2_backbone_out"]["backbone_fpn"]
            ]
            sam2_backbone_out["vision_pos_enc"] = [
                pos[image_index : image_index + 1]
                for pos in backbone_out_image_only["sam2_backbone_out"]["vision_pos_enc"]
            ]
            if backbone_out_image_only["sam2_backbone_out"].get("vision_features") is not None:
                sam2_backbone_out["vision_features"] = backbone_out_image_only[
                    "sam2_backbone_out"
                ]["vision_features"][image_index : image_index + 1]
            single_backbone_out["sam2_backbone_out"] = sam2_backbone_out

        for cache_key in (
            "_cached_img_feats",
            "_cached_img_pos_embeds",
            "_cached_vis_feat_sizes",
        ):
            single_backbone_out.pop(cache_key, None)

        return single_backbone_out

    def _build_autocast_context(self, dtype_name):
        if self.device.type != "cuda":
            return nullcontext()
        if dtype_name == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if dtype_name == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _encoder_autocast_context(self):
        return self._build_autocast_context(self.image_encoder_dtype)

    def _decoder_autocast_context(self):
        return self._build_autocast_context(self.inference_dtype)

    def _normalize_query_indices(self, query_indices):
        if query_indices is None:
            return torch.arange(self.num_queries, device=self.device, dtype=torch.long)
        if isinstance(query_indices, torch.Tensor):
            return query_indices.to(device=self.device, dtype=torch.long)
        return torch.tensor(query_indices, device=self.device, dtype=torch.long)

    def _stack_component_outputs(self, outputs):
        keys = outputs[0].keys()
        return {key: torch.stack([output[key] for output in outputs], dim=0) for key in keys}

    def _slice_batch_output(self, output, batch_index):
        if isinstance(output, dict):
            return {key: value[batch_index] for key, value in output.items()}
        return output[batch_index]

    def _cast_backbone_out_to_decoder_dtype(self, backbone_out):
        target_dtype = self._dtype_from_name(self.inference_dtype)
        if target_dtype is None:
            return backbone_out
        if self._dtype_from_name(self.image_encoder_dtype) == target_dtype:
            return backbone_out
        return self._cast_nested(backbone_out, target_dtype)

    def _cast_nested(self, value, target_dtype):
        if isinstance(value, dict):
            return {key: self._cast_nested(val, target_dtype) for key, val in value.items()}
        if isinstance(value, list):
            return [self._cast_nested(item, target_dtype) for item in value]
        if isinstance(value, tuple):
            return tuple(self._cast_nested(item, target_dtype) for item in value)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.to(dtype=target_dtype)
        return value

    def _dtype_from_name(self, dtype_name):
        if dtype_name == "fp32":
            return torch.float32
        if dtype_name == "bf16":
            return torch.bfloat16
        if dtype_name == "fp16":
            return torch.float16
        return None
