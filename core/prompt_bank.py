import hashlib
import os
import pickle
import re
from contextlib import nullcontext
from difflib import SequenceMatcher

import torch


class PromptBank:
    """Manages prompt expansion, indexing, and reusable text embeddings."""

    _text_embedding_cache = {}

    def __init__(
        self,
        processor,
        classname_path,
        device,
        expanded_prompt_pool_path=None,
        cache_text_embeddings=True,
        enable_expanded_prompt=False,
        inference_dtype="fp32",
    ):
        self.processor = processor
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.cache_text_embeddings = cache_text_embeddings
        self.enable_expanded_prompt = enable_expanded_prompt
        self.inference_dtype = inference_dtype

        query_words, query_idx = self.get_cls_idx(classname_path)
        print(f"Initial query words from {classname_path}: {len(query_words)}")
        for i, (word, idx) in enumerate(zip(query_words, query_idx[: len(query_words)])):
            print(f"  {i + 1}. {word} (index {idx})")

        self.expanded_prompt_pool = None
        print(f"Looking for expanded prompt pool at: {expanded_prompt_pool_path}")
        if enable_expanded_prompt and expanded_prompt_pool_path and os.path.exists(expanded_prompt_pool_path):
            print(f"Loading expanded prompt pool from {expanded_prompt_pool_path}")
            with open(expanded_prompt_pool_path, "rb") as f:
                self.expanded_prompt_pool = pickle.load(f)

            print(f"Loaded expanded prompt pool with {len(self.expanded_prompt_pool)} classes:")
            for class_name, variants in self.expanded_prompt_pool.items():
                print(f"  {class_name}: {variants}")

            expanded_query_words, expanded_query_idx = self._expand_query_words(
                query_words, query_idx
            )
            if len(expanded_query_words) <= len(query_words):
                fallback_pool = self._load_prompt_pool_from_exp_file(
                    classname_path, query_words, query_idx
                )
                if fallback_pool:
                    fallback_query_words, fallback_query_idx = self._expand_query_words(
                        query_words,
                        query_idx,
                        prompt_pool=fallback_pool,
                    )
                    if len(fallback_query_words) > len(expanded_query_words):
                        print(
                            "Loaded prompt pool did not expand query count; "
                            "falling back to the exp file for prompt expansion."
                        )
                        self.expanded_prompt_pool = fallback_pool
                        expanded_query_words = fallback_query_words
                        expanded_query_idx = fallback_query_idx
            query_words, query_idx = expanded_query_words, expanded_query_idx
            print(f"Using expanded prompt pool with {len(query_words)} query words:")
        else:
            if enable_expanded_prompt:
                fallback_pool = self._load_prompt_pool_from_exp_file(
                    classname_path, query_words, query_idx
                )
                if fallback_pool:
                    self.expanded_prompt_pool = fallback_pool
                    query_words, query_idx = self._expand_query_words(
                        query_words,
                        query_idx,
                        prompt_pool=fallback_pool,
                    )
                    print(
                        "Expanded prompt pool file unavailable; using prompts parsed "
                        f"from {classname_path.replace('.txt', '_exp.txt')}."
                    )
                    print(f"Using expanded prompt pool with {len(query_words)} query words:")
                elif expanded_prompt_pool_path:
                    print(f"Expanded prompt pool file not found: {expanded_prompt_pool_path}")
                    print(f"Using original {len(query_words)} query words without expansion:")
                else:
                    print("No expanded prompt pool path provided")
                    print(f"Using original {len(query_words)} query words without expansion:")
            elif expanded_prompt_pool_path:
                print(f"Expanded prompt pool file not found: {expanded_prompt_pool_path}")
                print(f"Using original {len(query_words)} query words without expansion:")
            else:
                print("No expanded prompt pool path provided")
                print(f"Using original {len(query_words)} query words without expansion:")

        self.query_words = query_words
        self.query_idx = torch.tensor(query_idx, dtype=torch.int64, device=self.device)
        for i, (word, idx) in enumerate(zip(self.query_words, self.query_idx)):
            print(f"  {i + 1:2d}. {word} (class {int(idx)})")

        self.num_queries = len(self.query_words)
        self.num_cls = int(self.query_idx.max().item()) + 1

        query_words_hash = hashlib.md5("|".join(self.query_words).encode()).hexdigest()
        self.cache_key = f"{classname_path}_{query_words_hash}_{str(device)}"

        if cache_text_embeddings and self.cache_key in PromptBank._text_embedding_cache:
            print(f"Loading cached text embeddings for {self.cache_key}")
            self.text_embeddings = PromptBank._text_embedding_cache[self.cache_key]
        else:
            print(f"Computing text embeddings for {len(self.query_words)} query words...")
            self.text_embeddings = self._compute_text_embeddings()
            if cache_text_embeddings:
                PromptBank._text_embedding_cache[self.cache_key] = {
                    key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in self.text_embeddings.items()
                }
                print(f"Cached text embeddings for {self.cache_key}")

    def _normalize_prompt_token(self, token):
        token = str(token).strip()
        token = token.replace("锛?, ", "").replace("_", " ").replace("-", " ")
        token = re.sub(r"\s+", " ", token)
        return token.lower().strip(" ,")

    def _split_prompt_tokens(self, prompt_text):
        prompt_text = str(prompt_text).strip()
        if not prompt_text:
            return []
        prompt_text = prompt_text.replace("锛?, ", "")
        tokens = [token.strip() for token in prompt_text.split(",") if token.strip()]
        return tokens if tokens else [prompt_text]

    def _match_class_index(self, candidate, class_to_aliases):
        normalized_candidate = self._normalize_prompt_token(candidate)
        if not normalized_candidate:
            return None

        for class_idx, aliases in class_to_aliases.items():
            normalized_aliases = {
                self._normalize_prompt_token(alias) for alias in aliases if alias
            }
            if normalized_candidate in normalized_aliases:
                return class_idx

        candidate_tokens = set(normalized_candidate.split())
        best_class_idx = None
        best_score = 0.0
        best_overlap = 0.0
        best_seq_ratio = 0.0
        for class_idx, aliases in class_to_aliases.items():
            for alias in aliases:
                normalized_alias = self._normalize_prompt_token(alias)
                alias_tokens = set(normalized_alias.split())
                token_overlap = len(candidate_tokens & alias_tokens) / max(
                    len(candidate_tokens | alias_tokens), 1
                )
                seq_ratio = SequenceMatcher(
                    None, normalized_candidate, normalized_alias
                ).ratio()
                score = 0.65 * seq_ratio + 0.35 * token_overlap
                if score > best_score:
                    best_class_idx = class_idx
                    best_score = score
                    best_overlap = token_overlap
                    best_seq_ratio = seq_ratio

        if best_score >= 0.55:
            return best_class_idx
        if best_overlap > 0.0 and best_seq_ratio >= 0.3:
            return best_class_idx
        return None

    def _load_prompt_pool_from_exp_file(self, classname_path, query_words, query_idx):
        exp_classname_path = classname_path.replace(".txt", "_exp.txt")
        if not os.path.exists(exp_classname_path):
            return None

        class_to_aliases = {}
        class_order = []
        for word, idx in zip(query_words, query_idx):
            class_idx = int(idx)
            if class_idx not in class_to_aliases:
                class_to_aliases[class_idx] = []
                class_order.append(class_idx)
            if word not in class_to_aliases[class_idx]:
                class_to_aliases[class_idx].append(word)

        prompt_pool = {}
        with open(exp_classname_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = self._split_prompt_tokens(line)
                if not parts:
                    continue

                class_idx = self._match_class_index(parts[0], class_to_aliases)
                if class_idx is None:
                    continue

                canonical_alias = class_to_aliases[class_idx][0]
                prompt_variants = prompt_pool.setdefault(
                    canonical_alias, [canonical_alias]
                )
                for part in parts[1:]:
                    cleaned_part = part.strip()
                    if (
                        cleaned_part
                        and cleaned_part not in prompt_variants
                        and cleaned_part not in class_to_aliases[class_idx]
                    ):
                        prompt_variants.append(cleaned_part)

        for class_idx in class_order:
            canonical_alias = class_to_aliases[class_idx][0]
            prompt_pool.setdefault(canonical_alias, [canonical_alias])

        return prompt_pool

    def _expand_query_words(self, query_words, query_idx, prompt_pool=None):
        expanded_query_words = []
        expanded_query_idx = []
        global_seen = set()
        prompt_pool = prompt_pool if prompt_pool is not None else self.expanded_prompt_pool

        class_to_aliases = {}
        class_order = []
        for word, idx in zip(query_words, query_idx):
            class_idx = int(idx)
            if class_idx not in class_to_aliases:
                class_to_aliases[class_idx] = []
                class_order.append(class_idx)
            if word not in class_to_aliases[class_idx]:
                class_to_aliases[class_idx].append(word)

        for class_idx in class_order:
            aliases = class_to_aliases[class_idx]
            local_candidates = []
            local_seen = set()

            def add_local(candidate):
                candidate = candidate.strip()
                if not candidate or candidate in local_seen or candidate in global_seen:
                    return
                local_candidates.append(candidate)
                local_seen.add(candidate)

            matched = False
            normalized_aliases = {
                self._normalize_prompt_token(alias) for alias in aliases if alias
            }
            for pool_key, prompt_variants in prompt_pool.items():
                pool_key_parts = self._split_prompt_tokens(pool_key)
                normalized_pool_key_parts = {
                    self._normalize_prompt_token(part)
                    for part in pool_key_parts
                    if part
                }
                matched_via_alias = any(
                    alias in normalized_pool_key_parts for alias in normalized_aliases
                )
                if not matched_via_alias:
                    matched_class_idx = self._match_class_index(
                        pool_key, class_to_aliases
                    )
                    if matched_class_idx != class_idx:
                        continue
                if matched_via_alias:
                    for alias in aliases:
                        add_local(alias)
                for prompt_variant in prompt_variants:
                    candidates = self._split_prompt_tokens(prompt_variant)
                    for candidate in candidates:
                        add_local(candidate)
                matched = True
                break

            if not matched:
                for alias in aliases:
                    add_local(alias)

            for candidate in local_candidates:
                expanded_query_words.append(candidate)
                expanded_query_idx.append(class_idx)
                global_seen.add(candidate)

        return expanded_query_words, expanded_query_idx

    def _compute_text_embeddings(self):
        with torch.no_grad():
            with self._autocast_context():
                text_outputs = self.processor.model.backbone.forward_text(
                    self.query_words,
                    device=self.device,
                )

        language_features = text_outputs.get("language_features")
        language_mask = text_outputs.get("language_mask")
        language_embeds = text_outputs.get("language_embeds")

        if language_features is None or language_mask is None:
            raise RuntimeError(
                "SAM3 forward_text did not return required language_features/language_mask"
            )

        print(
            "Computed text embeddings: "
            f"language_features {language_features.shape}, "
            f"language_mask {language_mask.shape}"
        )

        return {
            "language_features": language_features,
            "language_mask": language_mask,
            "language_embeds": language_embeds,
        }

    def _autocast_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        if self.inference_dtype == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.inference_dtype == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def build_backbone_out(
        self,
        backbone_out_image_only,
        start_idx,
        end_idx,
        image_repeats=1,
    ):
        query_indices = torch.arange(
            start_idx, end_idx, device=self.device, dtype=torch.long
        )
        return self.build_backbone_out_by_indices(
            backbone_out_image_only=backbone_out_image_only,
            query_indices=query_indices,
            image_repeats=image_repeats,
        )

    def build_backbone_out_by_indices(
        self,
        backbone_out_image_only,
        query_indices,
        image_repeats=1,
    ):
        backbone_out = dict(backbone_out_image_only)
        if not isinstance(query_indices, torch.Tensor):
            query_indices = torch.tensor(
                query_indices, device=self.device, dtype=torch.long
            )
        else:
            query_indices = query_indices.to(device=self.device, dtype=torch.long)

        language_features = self.text_embeddings["language_features"].index_select(
            1, query_indices
        )
        language_mask = self.text_embeddings["language_mask"].index_select(
            0, query_indices
        )
        language_embeds = self.text_embeddings.get("language_embeds")

        if image_repeats > 1:
            language_features = language_features.repeat(1, image_repeats, 1)
            language_mask = language_mask.repeat(image_repeats, 1)
            if language_embeds is not None:
                language_embeds = language_embeds.index_select(1, query_indices).repeat(
                    1, image_repeats, 1
                )
        elif language_embeds is not None:
            language_embeds = language_embeds.index_select(1, query_indices)

        backbone_out["language_features"] = language_features
        backbone_out["language_mask"] = language_mask
        if language_embeds is not None:
            backbone_out["language_embeds"] = language_embeds
        return backbone_out

    @staticmethod
    def get_cls_idx(path):
        with open(path, "r") as f:
            name_sets = f.readlines()

        class_names = []
        class_indices = []
        for idx, name_set in enumerate(name_sets):
            names_i = [item.strip() for item in name_set.split(",")]
            class_names += names_i
            class_indices += [idx for _ in range(len(names_i))]

        class_names = [item.replace("\n", "") for item in class_names]
        return class_names, class_indices

