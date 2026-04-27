# core/qwen_agent.py
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import json
import re
import os
import pickle
from difflib import SequenceMatcher
from config import Config


class QwenAgent:
    """Qwen3-VL Intelligent Agent - English Version"""
    
    def __init__(self, model_path=Config.QWEN_MODEL_PATH, device=None, dataset_name=None):
        print("Loading Qwen3-VL model...")
        
        # Get device string
        if device is None:
            self.device_str = Config.get_qwen_device()
        else:
            self.device_str = device
        
        # Store dataset name for dynamic prompt loading
        self.dataset_name = dataset_name
        
        print(f"  Model path: {model_path}")
        print(f"  Device: {self.device_str}")
        if self.dataset_name:
            print(f"  Dataset: {self.dataset_name}")
        
        try:
            # Get data type
            torch_dtype_str = Config.get_qwen_dtype()
            torch_dtype = getattr(torch, torch_dtype_str)
            
            print(f"  Model precision: torch.{torch_dtype_str}")
            
            # Load model
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=self.device_str if self.device_str != "cpu" else None,
                trust_remote_code=True
            )
            
            # If device is CPU, move model manually
            if self.device_str == "cpu":
                self.model = self.model.to("cpu")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            print(f"鉁?Qwen3-VL model loaded successfully")
            print(f"  Model device: {self.model.device}")
            
        except Exception as e:
            print(f"鉁?Model loading failed: {e}")
            print("Trying alternative loading methods...")
            self._load_with_fallback(model_path)
        
        self.model.eval()

    def _normalize_optimize_method(self, optimize_method):
        method = str(optimize_method).strip().lower()
        alias_map = {
            "preset": {"preset"},
            "guided": {"guided"},
            "adaptive": {"adaptive"},
            "hybrid": {"hybrid"},
            "hybrid_strict": {
                "hybrid_strict",
                "hybrid_str",
                "hybid_strict",
                "hybid_str",
                "hybrid-strict",
                "hybrid strict",
            },
        }

        for canonical_name, aliases in alias_map.items():
            if method in aliases:
                if method != canonical_name:
                    print(
                        f"Normalizing optimize_method from '{optimize_method}' "
                        f"to '{canonical_name}'."
                    )
                return canonical_name

        valid_methods = ", ".join(sorted(alias_map))
        raise ValueError(
            f"Unsupported optimize_method '{optimize_method}'. "
            f"Expected one of: {valid_methods}."
        )
    
    def _load_with_fallback(self, model_path):
        """Alternative loading methods"""
        try:
            print("Trying without device_map...")
            
            # Get data type
            torch_dtype_str = Config.get_qwen_dtype()
            torch_dtype = getattr(torch, torch_dtype_str)
            
            # Method 1: Load to CPU first then move
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=None,
                trust_remote_code=True
            )
            
            # Move to specified device
            if self.device_str != "cpu":
                self.model = self.model.to(self.device_str)
            
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            print("鉁?Alternative method 1 successful")
            
        except Exception as e:
            print(f"Alternative method 1 failed: {e}")
            
            try:
                print("Trying with lower precision...")
                # Method 2: Use FP32 to reduce memory
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True
                )
                
                # Move to specified device
                if self.device_str != "cpu":
                    self.model = self.model.to(self.device_str)
                
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                print("鉁?Alternative method 2 successful")
                
            except Exception as e:
                print(f"Alternative method 2 failed: {e}")
                raise RuntimeError("All loading methods failed")
    
    def analyze_scene(self, image_path, detail_level="high"):
        """
        Detailed scene analysis for full segmentation
        
        Args:
            image_path: Image path
            detail_level: Detail level (low, medium, high)
            
        Returns:
            Scene analysis result in English
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path
            
            # Select prompt based on detail level
            if detail_level == "high":
                prompt = """Please analyze this image in detail as a professional image analyst.
                
                Please describe in detail following this structure:
                
                1. Main Object Identification:
                   - List all visible main objects (at least 10)
                   - Describe each object's position, size, color, shape
                
                2. Scene Composition:
                   - Foreground, midground, background division
                   - Spatial layout and perspective
                
                3. Object Categories:
                   - Person category (e.g., person, child, adult)
                   - Vehicle category (e.g., car, bicycle, motorcycle)
                   - Building category (e.g., building, house, shop)
                   - Nature category (e.g., tree, plant, sky, cloud)
                   - Other objects (e.g., furniture, equipment, signs)
                
                4. Special Attention:
                   - Occlusion relationships
                   - Repeated objects
                   - Small or subtle objects
                
                Please answer in English. Description must be extremely detailed and specific."""
            elif detail_level == "medium":
                prompt = """Please analyze this image and list all visible objects and regions. Answer in English."""
            else:
                prompt = """Please briefly describe this image content. Answer in English."""
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Prepare inputs
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate analysis
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2000,  # Increase tokens
                    temperature=0.1,
                    do_sample=False
                )
            
            # Decode output
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Extract assistant response
            if "assistant" in generated_text:
                response = generated_text.split("assistant")[-1].strip()
            else:
                response = generated_text
            
            response = response.replace("```", "")
            return response.strip()
        
        except Exception as e:
            print(f"Error during scene analysis: {e}")
            return ""

    def _run_image_prompt(
        self,
        image_path,
        prompt,
        max_new_tokens=768,
        temperature=0.1,
        do_sample=False,
    ):
        """Run a single image + text prompt and return decoded assistant text."""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": str(prompt)},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        if "assistant" in generated_text:
            response = generated_text.split("assistant")[-1].strip()
        else:
            response = generated_text.strip()
        return response.replace("```", "").strip()

    def analyze_remote_sensing_with_context(
        self,
        image_path,
        context_text,
        language="zh",
    ):
        """
        Analyze a remote sensing image with structured SAM3 context.

        Args:
            image_path: image path or PIL image
            context_text: textual context including prompt strategies and segmentation summaries
            language: "zh" or "en"

        Returns:
            str: model-generated analysis
        """
        context_text = str(context_text or "").strip()
        if not context_text:
            return ""

        if str(language).lower().startswith("zh"):
            prompt = f"""你是一名服务于大数据实践赛的遥感智能评估助手。请结合输入图像，以及下面给出的 SAM3 分割结果摘要，对图像做场景化业务分析。

要求：
1. 用中文输出。
2. 不要评价提示词好坏，也不要讨论调参过程。
3. 重点解释分割结果说明了什么，以及它对所选业务场景有什么价值。
4. 必须结合面积占比、主导地物、细粒目标数量、边界复杂度或混合区域等线索展开分析。
5. 如果证据不足以支持强结论，要明确指出不确定性。
6. 输出为简洁的 Markdown，包含以下小标题：
   - 场景结论
   - 分割发现
   - 业务解读
   - 决策建议
   - 风险与局限

下面是 SAM3 结果摘要：
{context_text}
"""
        else:
            prompt = f"""You are a remote-sensing assessment assistant for a data-science competition. Use the image and the SAM3 segmentation context below to produce a scene-oriented business analysis.

Requirements:
1. Do not judge prompt quality or discuss prompt engineering.
2. Focus on what the segmentation reveals and why it matters for the selected application scenario.
3. Cover scene structure, dominant land cover, fine targets, decision value, and uncertainty.
4. Output in Markdown with sections:
   - Scenario Conclusion
   - Segmentation Findings
   - Business Interpretation
   - Decision Suggestions
   - Risks and Limits

SAM3 context:
{context_text}
"""

        try:
            return self._run_image_prompt(
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=900,
                temperature=0.15,
                do_sample=False,
            )
        except Exception as e:
            print(f"Error during contextual remote sensing analysis: {e}")
            return ""

    def _normalize_segmentation_prompt(self, prompt_text, max_words=3):
        cleaned = str(prompt_text).strip().lower()
        cleaned = cleaned.replace("_", " ").replace("-", " ")
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return ""
        words = cleaned.split()
        if not words:
            return ""
        return " ".join(words[:max_words])

    def _fallback_segmentation_prompts(
        self, user_instruction, max_prompts=8, max_words=3
    ):
        prompts = []
        for raw_chunk in re.split(r"[,;\n]+", str(user_instruction or "")):
            normalized = self._normalize_segmentation_prompt(
                raw_chunk, max_words=max_words
            )
            if normalized and normalized not in prompts:
                prompts.append(normalized)
            if len(prompts) >= max_prompts:
                break
        return prompts

    def generate_segmentation_prompts(
        self,
        image_path,
        user_instruction="",
        max_prompts=8,
        max_words=3,
    ):
        """
        Generate short segmentation prompts from an image and a user instruction.

        Returns:
            dict with keys:
                analysis: short textual summary
                prompts: list[str]
                raw_response: raw decoded model output
                error: optional error string
        """
        try:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path

            instruction = str(user_instruction or "").strip()
            if not instruction:
                instruction = (
                    "Segment the most relevant visible objects and regions in this image."
                )

            dataset_hint = ""
            if self.dataset_name:
                dataset_hint = (
                    f"Dataset context: {self.dataset_name}. "
                    "Favor prompts that match remote sensing imagery."
                )

            prompt = f"""You are preparing prompts for an image segmentation model.
{dataset_hint}
User instruction: {instruction}

Return only valid JSON using this schema:
{{"analysis":"one short sentence","prompts":["prompt one","prompt two"]}}

Rules:
- every prompt must describe visible image content
- every prompt must contain at most {max_words} words
- use lowercase English
- avoid punctuation except spaces
- avoid duplicates
- avoid numbering
- output at most {max_prompts} prompts
- prefer concrete object or region names useful for segmentation
"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False,
                )

            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]

            if "assistant" in generated_text:
                response = generated_text.split("assistant")[-1].strip()
            else:
                response = generated_text.strip()

            response = (
                response.replace("```json", "")
                .replace("```JSON", "")
                .replace("```", "")
                .strip()
            )

            parsed = None
            try:
                parsed = json.loads(response)
            except Exception:
                json_match = re.search(r"\{.*\}", response, flags=re.S)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                    except Exception:
                        parsed = None

            analysis = ""
            prompts = []
            if isinstance(parsed, dict):
                analysis = str(parsed.get("analysis", "")).strip()
                raw_prompts = parsed.get("prompts", [])
                if isinstance(raw_prompts, str):
                    raw_prompts = [raw_prompts]
                if isinstance(raw_prompts, (list, tuple)):
                    for raw_prompt in raw_prompts:
                        normalized = self._normalize_segmentation_prompt(
                            raw_prompt, max_words=max_words
                        )
                        if normalized and normalized not in prompts:
                            prompts.append(normalized)
                        if len(prompts) >= max_prompts:
                            break

            if not prompts:
                for raw_line in response.splitlines():
                    normalized = self._normalize_segmentation_prompt(
                        raw_line, max_words=max_words
                    )
                    if normalized and normalized not in prompts:
                        prompts.append(normalized)
                    if len(prompts) >= max_prompts:
                        break

            if not prompts:
                prompts = self._fallback_segmentation_prompts(
                    instruction,
                    max_prompts=max_prompts,
                    max_words=max_words,
                )

            if not prompts:
                prompts = ["object"]

            return {
                "analysis": analysis,
                "prompts": prompts[:max_prompts],
                "raw_response": response,
                "error": "",
            }
        except Exception as e:
            print(f"Error generating segmentation prompts: {e}")
            fallback_prompts = self._fallback_segmentation_prompts(
                user_instruction,
                max_prompts=max_prompts,
                max_words=max_words,
            )
            if not fallback_prompts:
                fallback_prompts = ["object"]
            return {
                "analysis": "",
                "prompts": fallback_prompts[:max_prompts],
                "raw_response": "",
                "error": str(e),
            }

    def generate_expanded_class_prompts(self, base_classes, image_paths, output_path, predefined_prompts=None, optimize_method="guided"):
        """
        Generate expanded class prompts with different optimization methods
        
        Args:
            base_classes: Original class names
            image_paths: Paths to sample images for analysis (not needed for preset mode)
            output_path: Path to save the generated prompt pool
            predefined_prompts: Predefined prompts to use instead of generating new ones
            optimize_method: Optimization method ("guided", "adaptive", "hybrid", "preset", "hybrid_strict")
        """
        optimize_method = self._normalize_optimize_method(optimize_method)
        print(f"Generating expanded class prompts for classes: {base_classes}")
        print(f"Optimization method: {optimize_method}")
        
        if optimize_method == "preset":
            # Use preset optimization based on class names (designed to match exp files)
            print("Using preset optimization based on class names")
            prompt_pool = self._apply_preset_optimization(base_classes)
        elif optimize_method == "guided":
            # Use guided optimization with predefined prompts
            if predefined_prompts is not None:
                print("Using guided optimization with predefined prompts")
                prompt_pool = self._apply_guided_optimization(base_classes, predefined_prompts)
            else:
                print("No predefined prompts provided, using adaptive optimization")
                prompt_pool = self._apply_adaptive_optimization(base_classes, image_paths)
        elif optimize_method == "adaptive":
            # Use adaptive optimization based on image analysis
            print("Using adaptive optimization based on image analysis")
            prompt_pool = self._apply_adaptive_optimization(base_classes, image_paths)
        elif optimize_method == "hybrid":
            # Combine both approaches
            print("Using hybrid optimization approach")
            if predefined_prompts is not None:
                # Apply guided optimization with adaptive refinement
                guided_pool = self._apply_guided_optimization(base_classes, predefined_prompts)
                prompt_pool = self._apply_adaptive_refinement(guided_pool, image_paths)
            else:
                # Fallback to adaptive if no predefined prompts
                prompt_pool = self._apply_adaptive_optimization(base_classes, image_paths)
        elif optimize_method == "hybrid_strict":
            print("Using hybrid_strict optimization approach")
            if predefined_prompts is not None:
                guided_pool = self._apply_guided_optimization(base_classes, predefined_prompts)
                refined_pool = self._apply_adaptive_refinement(guided_pool, image_paths)
                prompt_pool = self._apply_strict_constraint_projection(
                    refined_pool, predefined_prompts
                )
            else:
                print("No predefined prompts provided, falling back to adaptive optimization")
                prompt_pool = self._apply_adaptive_optimization(base_classes, image_paths)
        else:
            raise ValueError(f"Unsupported optimize_method: {optimize_method}")
        
        # If not in preset-like strict modes, limit each class to at most 3 prompts
        if optimize_method not in ["preset", "hybrid_strict"]:
            limited_prompt_pool = {}
            for class_name, prompts in prompt_pool.items():
                # Keep only the first 3 prompts for each class
                limited_prompt_pool[class_name] = prompts[:3]
            prompt_pool = limited_prompt_pool
        
        # Save the prompt pool
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(prompt_pool, f)
        
        print(f"Expanded prompt pool generated and saved to {output_path}")
        print(f"Prompt pool contains {len(prompt_pool)} classes")
        for class_name, prompts in prompt_pool.items():
            print(f"  {class_name}: {len(prompts)} prompts")
            print(f"    Prompts: {prompts}")
        
        return prompt_pool

    def _apply_strict_constraint_projection(self, refined_pool, constraints):
        """
        Project any refined prompt pool back onto the exact exp-file constraint set.

        This keeps the final serialized pkl aligned with the constraint file, while
        still allowing image-based refinement to happen internally for analysis/logging.
        """
        constrained_pool = {}
        for class_name, prompts in constraints.items():
            constrained_pool[class_name] = list(prompts)
        return constrained_pool

    def _normalize_prompt_token(self, token):
        token = str(token).strip()
        token = token.replace("锛?, ", "").replace("_", " ").replace("-", " ")
        token = re.sub(r"\s+", " ", token)
        return token.lower().strip(" ,")

    def _split_prompt_line(self, line):
        normalized_line = str(line).strip()
        if not normalized_line:
            return []
        normalized_line = normalized_line.replace("锛?, ", "").replace("閿?", ",")
        parts = [part.strip() for part in normalized_line.split(",") if part.strip()]
        return parts if parts else [normalized_line]

    def _resolve_local_config_path(self, *candidate_paths):
        for candidate_path in candidate_paths:
            if not candidate_path:
                continue
            normalized_path = os.path.normpath(candidate_path)
            directory = os.path.dirname(normalized_path) or "."
            basename = os.path.basename(normalized_path)
            if not os.path.isdir(directory):
                continue
            try:
                entries = {entry.lower(): entry for entry in os.listdir(directory)}
            except OSError:
                entries = {}
            actual_name = entries.get(basename.lower())
            if actual_name:
                return os.path.join(directory, actual_name)
            if os.path.exists(normalized_path):
                return normalized_path
        return None

    def _match_base_class_name(self, candidate, base_classes):
        if not base_classes:
            return candidate

        normalized_candidate = self._normalize_prompt_token(candidate)
        if not normalized_candidate:
            return candidate

        normalized_lookup = {}
        for base_class in base_classes:
            normalized_lookup.setdefault(
                self._normalize_prompt_token(base_class), base_class
            )

        exact_match = normalized_lookup.get(normalized_candidate)
        if exact_match is not None:
            return exact_match

        candidate_tokens = set(normalized_candidate.split())
        best_match = candidate
        best_score = 0.0
        best_overlap = 0.0
        best_seq_ratio = 0.0
        for base_class in base_classes:
            normalized_base = self._normalize_prompt_token(base_class)
            base_tokens = set(normalized_base.split())
            token_overlap = len(candidate_tokens & base_tokens) / max(
                len(candidate_tokens | base_tokens), 1
            )
            seq_ratio = SequenceMatcher(
                None, normalized_candidate, normalized_base
            ).ratio()
            score = 0.65 * seq_ratio + 0.35 * token_overlap
            if score > best_score:
                best_match = base_class
                best_score = score
                best_overlap = token_overlap
                best_seq_ratio = seq_ratio

        if best_score >= 0.55:
            return best_match
        if best_overlap > 0.0 and best_seq_ratio >= 0.3:
            return best_match
        return candidate

    def _resolve_exp_file_path(self, dataset_name):
        dataset_key = str(dataset_name or "").lower()
        candidate_paths = [f"./configs/cls_{dataset_key}_exp.txt"]
        if dataset_key == "isaid":
            candidate_paths.insert(0, "./configs/cls_iSAID_exp.txt")
        return self._resolve_local_config_path(*candidate_paths)

    def _load_exp_file_prompt_mapping(self, exp_file_path, base_classes):
        prompt_mapping = {}
        if not exp_file_path or not os.path.exists(exp_file_path):
            return prompt_mapping

        with open(exp_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = self._split_prompt_line(line)
                if not parts:
                    continue

                matched_base = self._match_base_class_name(parts[0], base_classes)
                prompts = prompt_mapping.setdefault(matched_base, [matched_base])
                for prompt in parts[1:]:
                    cleaned_prompt = prompt.strip()
                    if cleaned_prompt and cleaned_prompt not in prompts:
                        prompts.append(cleaned_prompt)

        return prompt_mapping

    def _infer_dataset_name_from_classes(self, base_classes, match_ratio=0.5):
        dataset_indicators = {
            "loveda": [
                "background",
                "building",
                "road",
                "water",
                "barren",
                "forest",
                "agricultural",
            ],
            "potsdam": ["imprev", "building", "low_vegetation", "tree", "car"],
            "vaihingen": [
                "impervious_surface",
                "building",
                "low_vegetation",
                "tree",
                "car",
                "clutter",
            ],
            "isaid": [
                "ship",
                "storage_tank",
                "baseball_diamond",
                "tennis_court",
                "basketball_court",
                "Ground_Track_Field",
                "Bridge",
                "Large_Vehicle",
                "Small_Vehicle",
                "Helicopter",
                "Swimming_pool",
                "Roundabout",
                "Soccer_ball_field",
                "Plane",
                "Harbor",
            ],
        }

        for dataset_name, dataset_classes in dataset_indicators.items():
            matches = len(set(base_classes).intersection(set(dataset_classes)))
            if matches >= len(dataset_classes) * match_ratio:
                return dataset_name
        return None

    def _load_preset_prompt_pool(self, base_classes):
        prompt_pool = {base_class: [base_class] for base_class in base_classes}

        target_dataset = None
        if self.dataset_name:
            dataset_name = self.dataset_name.lower()
            if "vaihingen" in dataset_name or "isprs" in dataset_name:
                target_dataset = "vaihingen"
            elif "potsdam" in dataset_name:
                target_dataset = "potsdam"
            elif "loveda" in dataset_name:
                target_dataset = "loveda"
            elif "isaid" in dataset_name:
                target_dataset = "isaid"

        if target_dataset is None:
            target_dataset = self._infer_dataset_name_from_classes(
                base_classes, match_ratio=0.5
            )

        exp_file_path = self._resolve_exp_file_path(target_dataset)
        if not exp_file_path or not os.path.exists(exp_file_path):
            print(
                f"Exp file not found for dataset '{target_dataset}' at {exp_file_path}. "
                "Using base classes."
            )
            return prompt_pool

        print(f"Loading constraints from: {exp_file_path}")
        try:
            exp_file_content = self._load_exp_file_prompt_mapping(
                exp_file_path, base_classes
            )
        except Exception as e:
            print(f"Error loading exp file {exp_file_path}: {e}")
            return prompt_pool

        for base_class in base_classes:
            exp_prompts = exp_file_content.get(base_class)
            if exp_prompts:
                prompt_pool[base_class] = list(exp_prompts)
            else:
                print(f"No constraints found for class '{base_class}', using base class only.")

        return prompt_pool
    
    def _load_exp_prompts(self, base_classes):
        """
        Load expanded prompts from .exp file based on dataset name inferred from base_classes.
        Expected file format: Each line contains a base class followed by its expanded prompts,
        e.g., "building,house,structure" or "background锛宑lutter".
        """
        prompt_pool = {}
        
        # Determine the dataset name based on the base classes provided
        dataset_indicators = {
            'loveda': ['background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural'],
            'potsdam': ['imprev', 'building', 'low_vegetation', 'tree', 'car'],
            'vaihingen': ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter'],
            'isaid': ['ship', 'storage_tank', 'baseball_diamond', 'tennis_court', 'basketball_court', 
                     'Ground_Track_Field', 'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter', 
                     'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'Plane', 'Harbor']
        }
        
        dataset_name = None
        for ds_name, ds_classes in dataset_indicators.items():
            # Check if majority of our base classes match this dataset's typical classes
            if len(set(base_classes).intersection(set(ds_classes))) >= len(ds_classes) / 3:  # At least 1/3 match
                dataset_name = ds_name
                break
        
        if dataset_name is None:
            # Fallback: try to infer from class names
            if 'agricultural' in base_classes or 'barren' in base_classes:
                dataset_name = 'loveda'
            elif 'imprev' in base_classes or 'low_vegetation' in base_classes:
                dataset_name = 'potsdam'
            elif 'impervious_surface' in base_classes or 'clutter' in base_classes:
                dataset_name = 'vaihingen'
            elif any(c in ['ship', 'storage_tank', 'tennis_court'] for c in base_classes):
                dataset_name = 'isaid'
        
        if dataset_name is None:
            print("Warning: Could not determine dataset name for preset optimization. Using base classes only.")
            for base_class in base_classes:
                prompt_pool[base_class] = [base_class]
            return prompt_pool

        # Construct path for the exp file based on dataset name
        exp_file_path = self._resolve_exp_file_path(dataset_name)
        
        if exp_file_path and os.path.exists(exp_file_path):
            print(f"Loading expanded prompts from: {exp_file_path}")
            try:
                with open(exp_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Parse the file - expecting format like:
                # background锛宑lutter
                # building,house
                # road
                # water,water pool,lake,river
                
                temp_mapping = {}
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Handle both comma and Chinese comma separators
                    parts = []
                    if ',' in line:
                        parts = [p.strip() for p in line.split(',') if p.strip()]
                    elif '，' in line or '锛?' in line:  # Chinese comma
                        normalized_line = line.replace('锛?', '，')
                        parts = [p.strip() for p in normalized_line.split('，') if p.strip()]
                    else:
                        # Just one class name
                        parts = [line.strip()]
                    
                    if parts:
                        # First part is the key (base class)
                        key = parts[0]
                        # Remaining parts are the expansions
                        vals = parts[1:] if len(parts) > 1 else []
                        
                        # Add the key itself if not in vals
                        if key not in vals:
                            vals.insert(0, key)
                        
                        temp_mapping[key] = vals

                # Map to base_classes
                for base_class in base_classes:
                    found_prompts = []
                    # Check for exact match in keys
                    if base_class in temp_mapping:
                        found_prompts = temp_mapping[base_class]
                    else:
                        # Try to find closest match
                        for key, prompts in temp_mapping.items():
                            if key.lower() == base_class.lower():
                                found_prompts = prompts
                                break
                    
                    if found_prompts:
                        prompt_pool[base_class] = found_prompts
                    else:
                        prompt_pool[base_class] = [base_class]
                        
            except Exception as e:
                print(f"Error loading exp file {exp_file_path}: {e}")
                for base_class in base_classes:
                    prompt_pool[base_class] = [base_class]
        else:
            print(f"Exp file not found for dataset '{dataset_name}' at {exp_file_path}. Using base classes.")
            for base_class in base_classes:
                prompt_pool[base_class] = [base_class]
                
        return prompt_pool

    def _apply_preset_optimization(self, base_classes):
        """Apply preset optimization that strictly follows exp file constraints without additional variations"""
        print(f"Applying preset optimization with strict exp file constraints")
        return self._load_preset_prompt_pool(base_classes)
        
        # Load exp file constraints based on the dataset name
        exp_file_content = {}
        
        if self.dataset_name:
            # Try to match dataset name variations
            dataset_name = self.dataset_name.lower()
            if 'vaihingen' in dataset_name or 'isprs' in dataset_name:
                exp_file_path = "./configs/cls_vaihingen_exp.txt"
            elif 'potsdam' in dataset_name:
                exp_file_path = "./configs/cls_potsdam_exp.txt"
            elif 'loveda' in dataset_name:
                exp_file_path = "./configs/cls_loveda_exp.txt"
            elif 'isaid' in dataset_name:
                exp_file_path = "./configs/cls_isaid_exp.txt"
            else:
                # Fallback to trying to identify based on base_classes content
                dataset_indicators = {
                    'loveda': ['background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural'],
                    'potsdam': ['imprev', 'building', 'low_vegetation', 'tree', 'car'],
                    'vaihingen': ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter'],
                    'isaid': ['ship', 'storage_tank', 'baseball_diamond', 'tennis_court', 'basketball_court', 
                             'Ground_Track_Field', 'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter', 
                             'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'Plane', 'Harbor']
                }
                
                identified_dataset = None
                for ds_name, ds_classes in dataset_indicators.items():
                    # Check if majority of our base classes match this dataset's typical classes
                    matches = len(set(base_classes).intersection(set(ds_classes)))
                    if matches >= len(ds_classes) * 0.5:  # At least 50% match
                        identified_dataset = ds_name
                        break
                
                if identified_dataset:
                    exp_file_path = f"./configs/cls_{identified_dataset}_exp.txt"
                else:
                    exp_file_path = None
                    
            if exp_file_path and os.path.exists(exp_file_path):
                print(f"Loading constraints from: {exp_file_path}")
                try:
                    with open(exp_file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Handle both comma and Chinese comma separators
                        parts = []
                        if ',' in line:
                            parts = [p.strip() for p in line.split(',') if p.strip()]
                        elif '，' in line or '锛?' in line:  # Chinese comma
                            normalized_line = line.replace('锛?', '，')
                            parts = [p.strip() for p in normalized_line.split('，') if p.strip()]
                        else:
                            # Just one class name
                            parts = [line.strip()]
                        
                        if parts:
                            # First part is the key (base class)
                            key = parts[0]
                            # Remaining parts are the expansions
                            vals = parts[1:] if len(parts) > 1 else []
                            
                            # Add the key itself if not in vals
                            if key not in vals:
                                vals.insert(0, key)
                            
                            exp_file_content[key] = vals
                
                except Exception as e:
                    print(f"Error loading exp file {exp_file_path}: {e}")
        
        # Create prompt pool with strict adherence to exp file constraints
        prompt_pool = {}
        
        # Process each base class with exp file constraints
        for base_class in base_classes:
            # Start with the base class
            constrained_prompts = [base_class]
            
            # Apply strict constraints from exp file without additional variations
            if base_class in exp_file_content:
                # Only use the exact expansions from the exp file
                exp_prompts = exp_file_content[base_class]
                
                for exp_prompt in exp_prompts:
                    if exp_prompt != base_class and exp_prompt not in constrained_prompts:
                        constrained_prompts.append(exp_prompt)
            else:
                # If no constraints available for this class, just use base class
                print(f"No constraints found for class '{base_class}', using base class only.")
        
            prompt_pool[base_class] = constrained_prompts
        
        return prompt_pool
    
    def _simulate_qwen3_inference(self, base_class):
        """
        Simulate Qwen3 inference process for a base class.
        In a real scenario, this would involve analyzing images and generating
        multiple descriptive variations for the class.
        """
        # Start with the base class
        inferred_prompts = [base_class]
        
        # Apply general expansion logic to simulate Qwen3 inference
        # This could include generating synonyms, descriptive phrases, etc.
        general_expansions = {
            'building': ['structure', 'edifice', 'construction'],
            'road': ['street', 'path', 'pavement', 'way'],
            'tree': ['wood', 'foliage', 'canopy'],
            'car': ['vehicle', 'automobile', 'auto'],
            'water': ['lake', 'river', 'pond'],
            'grass': ['lawn', 'turf', 'vegetation'],
            'clutter': ['background', 'miscellaneous']
        }
        
        if base_class in general_expansions:
            inferred_prompts.extend(general_expansions[base_class])
        
        return inferred_prompts
    
    def _apply_constraints_to_inferred_prompts(self, base_class, inferred_prompts, constraints):
        """
        Apply constraints from exp file to refine the inferred prompts.
        This simulates the "multi-return, addition/subtraction, replacement" logic.
        """
        # Start with the original inferred prompts
        refined_prompts = list(inferred_prompts)
        
        # Apply constraint processing logic
        for constraint in constraints:
            if constraint != base_class:
                # Add constraint to the refined prompts if not already present
                if constraint not in refined_prompts:
                    refined_prompts.append(constraint)
                
                # Apply variations based on the constraint
                variations = self._generate_constraint_variations(constraint)
                for variation in variations:
                    if variation not in refined_prompts:
                        refined_prompts.append(variation)
        
        return refined_prompts
    
    def _generate_constraint_variations(self, constraint):
        """
        Generate variations of a constraint to enrich the prompt pool.
        """
        variations = []
        
        # Add a "visual descriptor" variant if it makes sense
        if constraint in ['building', 'car', 'tree', 'road', 'grass', 'water']:
            visual_variants = [
                f"visual {constraint}",
                f"{constraint} object",
                f"{constraint} instance"
            ]
            variations.extend(visual_variants)
        
        # Add a "contextual" variant
        context_variants = [
            f"{constraint} in scene",
            f"{constraint} element"
        ]
        variations.extend(context_variants)
        
        return variations
    
    def _apply_guided_optimization(self, base_classes, predefined_prompts):
        """Apply guided optimization using predefined prompts"""
        prompt_pool = {}
        
        for base_class in base_classes:
            if base_class in predefined_prompts:
                # Use predefined prompts for this class
                prompt_pool[base_class] = predefined_prompts[base_class]
            else:
                # If no predefined prompt exists for this class, use the base class itself
                prompt_pool[base_class] = [base_class]
        
        return prompt_pool
    
    def _apply_adaptive_optimization(self, base_classes, image_paths):
        """Apply adaptive optimization by analyzing images"""
        prompt_pool = {}
        
        # Analyze a sample of images to generate class-specific prompts
        for base_class in base_classes:
            print(f"Adaptively optimizing prompts for class: {base_class}")
            
            # Collect descriptive terms for this class from multiple images
            class_prompts = set([base_class])  # Start with the base class name
            
            # Process first few images to gather descriptions
            for img_path in image_paths[:5]:  # Limit to first 5 images for efficiency
                try:
                    # Analyze the scene
                    analysis = self.analyze_scene(img_path, detail_level="high")
                    
                    # Extract relevant descriptions for the current class
                    analysis_lower = analysis.lower()
                    
                    # Simple keyword extraction based on class name
                    words = analysis_lower.split()
                    
                    # Find relevant descriptive terms near class name mentions
                    for i, word in enumerate(words):
                        if base_class.lower() in word or word in base_class.lower().split():
                            # Add surrounding context words as potential prompts
                            start = max(0, i-2)
                            end = min(len(words), i+3)
                            context = ' '.join(words[start:end])
                            
                            # Clean up the context to make it suitable as a prompt
                            cleaned_context = self._clean_prompt(context, base_class)
                            if cleaned_context:
                                class_prompts.add(cleaned_context)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            # Convert set to list and store
            prompt_pool[base_class] = list(class_prompts)
        
        return prompt_pool
    
    def _apply_adaptive_refinement(self, prompt_pool, image_paths):
        """Apply adaptive refinement to an existing prompt pool"""
        refined_pool = {}
        
        for class_name, prompts in prompt_pool.items():
            print(f"Refining prompts for class: {class_name}")
            
            refined_prompts = set(prompts)
            
            # Analyze images to refine the prompts
            for img_path in image_paths[:3]:  # Use fewer images for refinement
                try:
                    # Create targeted analysis for this class
                    prompt = f"Focus specifically on instances of '{class_name}' in this image. " \
                             f"Describe their appearance, characteristics, and how to distinguish them from other objects."
                    
                    # Build messages
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": Image.open(img_path).convert("RGB")},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    
                    text = self.processor.apply_chat_template(
                        messages, 
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    # Prepare inputs
                    inputs = self.processor(
                        text=text,
                        images=Image.open(img_path).convert("RGB"),
                        return_tensors="pt"
                    )
                    
                    # Move inputs to model device
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    # Generate analysis
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=500,
                            temperature=0.1,
                            do_sample=False
                        )
                    
                    # Decode output
                    generated_text = self.processor.batch_decode(
                        generated_ids, 
                        skip_special_tokens=True
                    )[0]
                    
                    # Extract assistant response
                    if "assistant" in generated_text:
                        response = generated_text.split("assistant")[-1].strip()
                    else:
                        response = generated_text
                    
                    # Extract descriptive terms from response
                    words = response.lower().split()
                    for i, word in enumerate(words):
                        if class_name.lower() in word or word in class_name.lower().split():
                            start = max(0, i-2)
                            end = min(len(words), i+3)
                            context = ' '.join(words[start:end])
                            cleaned_context = self._clean_prompt(context, class_name)
                            if cleaned_context and len(cleaned_context) > len(class_name):
                                refined_prompts.add(cleaned_context)

                except Exception as e:
                    print(f"Error refining for {img_path}: {e}")
                    continue
            
            refined_pool[class_name] = list(refined_prompts)
        
        return refined_pool
    
    def _clean_prompt(self, raw_text, base_class):
        """Clean up raw text to form a suitable prompt"""
        # Remove excessive punctuation and clean up
        cleaned = re.sub(r'[^\w\s,.-]', ' ', raw_text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Ensure it's not too long and contains meaningful information
        if len(cleaned) < 2 or len(cleaned) > 50:
            return ""
        
        # Make sure it relates to the base class somehow
        if base_class.lower() not in cleaned.lower():
            # Try to ensure we have something related to the class
            return f"{base_class} in {cleaned}"
        
        return cleaned
    
    def count_objects_by_prompt_pool(self, image_path, prompt_pool):
        """
        Count objects in an image based on the prompt pool
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path
            
            # Count objects for each class in the prompt pool
            counts = {}
            
            for class_name, prompts in prompt_pool.items():
                count = 0
                for prompt in prompts:
                    # Create a counting prompt for this specific class
                    counting_prompt = f"Count the number of {prompt} in this image. Respond with only the number."
                    
                    # Build messages
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": counting_prompt}
                            ]
                        }
                    ]
                    
                    text = self.processor.apply_chat_template(
                        messages, 
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    # Prepare inputs
                    inputs = self.processor(
                        text=text,
                        images=image,
                        return_tensors="pt"
                    )
                    
                    # Move inputs to model device
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    # Generate analysis
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=20,
                            temperature=0.1,
                            do_sample=False
                        )
                    
                    # Decode output
                    generated_text = self.processor.batch_decode(
                        generated_ids, 
                        skip_special_tokens=True
                    )[0]
                    
                    # Extract number from response
                    response = generated_text.split("assistant")[-1].strip()
                    numbers = re.findall(r'\d+', response)
                    if numbers:
                        count += int(numbers[0])
                
                counts[class_name] = count
            
            return counts
        
        except Exception as e:
            print(f"Error during object counting: {e}")
            return {}

    def generate_controlled_analysis(self, image_path, control_module_enabled=False, predefined_prompts=None):
        """
        Perform image analysis with optional control module
        """
        if control_module_enabled and predefined_prompts is not None:
            # Use controlled analysis with predefined prompts
            return self._controlled_analysis(image_path, predefined_prompts)
        else:
            # Standard analysis
            return self.analyze_scene(image_path)
    
    def _controlled_analysis(self, image_path, predefined_prompts):
        """
        Internal method for controlled analysis
        """
        # This would implement logic to guide Qwen3 using predefined prompts
        # For now, returning a structured response based on predefined prompts
        result = {
            "scene_analysis": self.analyze_scene(image_path),
            "available_classes": list(predefined_prompts.keys()),
            "prompt_variations": predefined_prompts
        }
        return result

