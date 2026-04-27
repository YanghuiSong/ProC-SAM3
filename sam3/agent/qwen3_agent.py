#!/usr/bin/env python3
"""
Qwen3 Agent for generating segmentation prompts based on image content
"""

import torch
import re
from transformers import AutoModelForImageTextToText, AutoTokenizer
from typing import List


class Qwen3Agent:
    """
    Qwen3 Agent for generating segmentation prompts based on image content
    """
    
    def __init__(self, model_path: str, device: str = "cuda:0",  # 默认改为cuda:0
                 quantization: bool = True):
        """
        Initialize Qwen3 Agent
        
        Args:
            model_path: Path to the local Qwen3 model
            device: Device to run the model on (e.g., "cuda:0", "cuda:1", "cpu")
            quantization: Whether to use 8-bit quantization to reduce memory usage
        """
        self.device = device
        print(f"Loading Qwen3 model from: {model_path}")
        print(f"Using device: {device}")
        print(f"Using quantization: {quantization}")
        
        # Load the local model and tokenizer using AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Use quantization to reduce memory usage if requested
        if quantization and device.startswith('cuda'):
            # Import bitsandbytes for quantization
            try:
                import bitsandbytes as bnb
                # Use 4-bit quantization to reduce memory usage even more
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path, 
                    quantization_config=bnb_config,
                    device_map=device,  # 使用指定的GPU设备
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                ).eval()
                print("Model loaded with 4-bit quantization")
            except ImportError:
                print("bitsandbytes not available, loading with CPU offload")
                # Fall back to CPU offloading
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,  # Use float16 to reduce memory
                    trust_remote_code=True,
                    device_map="auto",  # Automatically distribute across devices
                    low_cpu_mem_usage=True,
                    # Offload some layers to CPU to save GPU memory
                    offload_folder="./offload",
                    offload_state_dict=True,
                ).eval()
        else:
            # Load normally without quantization but with memory optimizations
            # Use auto device map to potentially utilize CPU for some layers
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, 
                torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
                trust_remote_code=True,
                device_map=device,  # 明确使用指定设备
                low_cpu_mem_usage=True,
                # Allow offloading to save GPU memory
                offload_folder="./offload" if torch.cuda.is_available() else None,
                offload_state_dict=True,
            ).eval()
        
        # Move model to specified device if not already handled by device_map
        if device != "auto":  # device_map为"auto"时不需要手动移动
            self.model = self.model.to(self.device)
    
    def generate_prompts_for_class(self, class_name: str, num_alternatives: int = 5) -> List[str]:
        """
        Generate alternative visual words for a given class
        
        Args:
            class_name: The target class name
            num_alternatives: Number of alternative prompts to generate
            
        Returns:
            List of alternative prompts
        """
        
        # Provide specialized examples for challenging classes
        if class_name == 'impervious_surface':
            example_words = "paved, concrete, asphalt, urban, developed, built-up, hardened, artificial, sealed, non-permeable, roadway, rooftop"
        elif class_name == 'low_vegetation':
            example_words = "grass, lawn, meadow, turf, herbaceous, groundcover, pasture, field, greensward, vegetated, grassy, short-vegetation"
        elif class_name == 'clutter':
            example_words = "mixed, debris, irregular, heterogeneous, noise, scattered, background, unclear, transition, complex, varied, fragmented"
        elif class_name == 'building':
            example_words = "house, home, cottage, residential, dwelling, garden, yard, private, suburban, detached, roof, gable, chimney, facade, brick, tile, structure, edifice, construction, architecture, facility"
        elif class_name == 'tree':
            example_words = "sparse, isolated, single, individual, standalone, bare_tree, branch, trunk, limb, dormant, leafless, winter, deciduous, complex, entangled, interlacing, rough, bark, gnarled, conical, oval, spreading, rounded"
        elif class_name == 'car':
            example_words = "vehicle, automobile, truck, van, motorcar, transport"
        elif class_name == 'grass':
            example_words = "lawn, meadow, pasture, turf, greensward, grassland"
        elif class_name == 'road':
            example_words = "street, path, lane, pavement, thoroughfare, roadway"
        else:
            example_words = "synonym1, synonym2, synonym3"
        
        # Create a text-only prompt for prompt generation with better instructions for SAM3
        prompt = (
            f"<|system|>\nYou are a helpful assistant that generates segmentation prompts for the SAM3 model.</s>\n"
            f"<|user|>\n"
            f"You are generating segmentation prompts for the SAM3 (Segment Anything Model 3) model.\n\n"
            f"Class: {class_name}\n\n"
            f"Generate {min(num_alternatives, 5)} alternative visual words for this class.\n\n"
            f"Rules:\n"
            f"- Keep each prompt to 1-2 words maximum (absolutely no more than 2 words)\n"
            f"- Focus on simple synonyms that describe the same object\n"
            f"- Use direct synonyms only, avoid descriptive phrases\n"
            f"- Examples for '{class_name}': {example_words}\n"
            f"- Return comma separated words only, nothing else.</s>\n"
            f"<|assistant|>\n"
        )
        
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,  # Further reduce token count to save memory
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,  # Prevent warning
                # Additional memory-saving options
                use_cache=True,
            )
        
        # Clear GPU cache to free memory
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        # Decode the response - only get the newly generated part
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        if not response:
            # Fallback if generation fails
            return [class_name]
        
        # Parse the response to extract individual prompts
        # Expected format: "word1, word2, word3, ..."
        # Split by comma and clean up each prompt
        raw_prompts = [p.strip() for p in response.split(",")]
        
        # Clean up prompts to remove any system text that might have leaked through
        cleaned_prompts = []
        for p in raw_prompts:
            # Remove any text that looks like system/user/assistant markers
            p = re.sub(r'<\|.*?\|>', '', p)  # Remove <|system|>, <|user|>, <|assistant|>
            
            # Split by newlines and take the first part (before any new interactions)
            p = p.split('\n')[0].strip()
            
            # Remove any remaining special tokens or phrases
            p = re.sub(r'Generate \d+ alternative visual words for this class.*', '', p, flags=re.DOTALL)
            p = re.sub(r'Rules:.*', '', p, flags=re.DOTALL)
            p = re.sub(r'Return.*', '', p, flags=re.DOTALL)
            
            p = p.strip()
            
            # Only add non-empty prompts that are reasonable (1-2 words and meaningful)
            if p and p != class_name and 1 <= len(p.split()) <= 2 and p not in cleaned_prompts:
                # Additional cleaning to ensure we only have simple terms
                p = p.lower()  # Convert to lowercase to normalize
                cleaned_prompts.append(p)
        
        # Limit to num_alternatives and add original class name
        prompts = cleaned_prompts[:num_alternatives-1]  # -1 because we'll add the original class name
        prompts.insert(0, class_name)  # Add original class name first
        
        return prompts

    def generate_count_and_descriptions(self, image_path: str, class_names: List[str], dataset_type: str = "auto") -> dict:
        """
        Generate object counts and hierarchical descriptions for each class in the image
        with structured short phrases that are CLIP-friendly (1-2 words).
        
        Args:
            image_path: Path to the image file
            class_names: List of class names to detect (used for class constraint)
            dataset_type: Type of dataset ("potsdam", "vaihingen", "loveda", "isaid", or "auto")
            
        Returns:
            A dictionary with class names as keys and counts/descriptions as values.
            Each entry contains:
            - count: Estimated number of instances
            - descriptions: List of visual prompts including baseline + variants
            - instance_features: Instance-level visual features (for SAM3 instance head)
            - semantic_features: Semantic-level features (for SAM3 semantic head)
            - confidence: Confidence score for presence (for SAM3 presence head)
        """
        
        # Detect dataset type if set to auto
        if dataset_type == "auto":
            image_lower = image_path.lower()
            if "potsdam" in image_lower or "pots" in image_lower:
                dataset_type = "potsdam"
            elif "vaihingen" in image_lower or "vh" in image_lower:
                dataset_type = "vaihingen"
            elif "loveda" in image_lower or "love" in image_lower:
                dataset_type = "loveda"
            elif "isaid" in image_lower or "isa" in image_lower:
                dataset_type = "isaid"
            else:
                # Default to vaihingen as it seems to work better overall
                dataset_type = "vaihingen"

        # Define general and instance class types for remote sensing
        general_classes = {'background', 'clutter', 'impervious_surface', 'low_vegetation'}  # General/area classes
        instance_classes = set(class_names) - general_classes  # Specific object classes
        
        # Parse class names to determine their types
        parsed_classes = {}
        for name in class_names:
            if name in general_classes:
                parsed_classes[name] = 'general'
            else:
                parsed_classes[name] = 'instance'
        
        # Format class names for the prompt
        class_list_str = ", ".join(class_names)
        
        # Dataset-specific context
        if dataset_type == "potsdam":
            # Potsdam dataset context - suburban/rural areas
            context_desc = (
                f"POTSDEM DATASET CONTEXT (SUBURBAN/RURAL):\n"
                f"- View perspective: Nadir (top-down) view over suburban areas\n"
                f"- Characteristics in Potsdam imagery:\n"
                f"  • Trees are more sparsely distributed and appear as isolated individuals (often along roads or in yards)\n"
                f"  • Roads may be dirt, gravel or older paved surfaces\n"
                f"  • Vegetation appears more natural and less manicured\n"
                f"  • Buildings often surrounded by gardens, not dense urban\n"
                f"  • More diverse textures due to mix of urban and rural elements\n"
                f"  • SEASONAL EFFECTS: Image may show winter conditions with leafless trees, brown/dry vegetation\n\n"
                
                f"ADDITIONAL POTSDEM CLASS-SPECIFIC HINTS:\n"
                f"- For TREE: Use precise terms like 'sparse', 'isolated', 'single', 'small', 'individual', 'standalone'\n"
                f"  AND SEASONAL TERMS: 'bare_tree', 'winter_tree', 'leafless_tree', 'branch_structure', 'tree_skeleton', 'dormant_tree'\n"
                f"  AND STRUCTURE TERMS: 'isolated_trunk', 'scattered_tree', 'single_tree', 'branch_structure', 'tree_skeleton', 'wooden_frame'\n"
                f"- For LOW_VEGETATION: Use 'sparse', 'natural', 'overgrown', 'wild', 'grass', 'dry_grass', 'brown_grass', 'dormant_grass', 'winter_ground_cover'\n"
                f"  AND WINTER-SPECIFIC: 'dormant_grass', 'winter_ground_cover', 'browned_vegetation', 'sparse_grass', 'dry_grassland', 'barren_field'\n"
                f"- FOR IMPERVIOUS_SURFACE: Use 'dirt', 'gravel', 'stone', 'unpaved', 'rough', 'soil', 'rural'\n"
                f"  AND ROAD-SPECIFIC TERMS: 'dirt_road', 'earthen_path', 'unpaved_trail', 'rural_pathway', 'country_track', 'field_path'\n"
                f"- FOR BUILDING: Use 'house', 'home', 'cottage', 'residential', 'dwelling', 'garden'\n"
                f"  AND RURAL-SPECIFIC: 'residential_roof', 'house_structure', 'rural_dwelling'\n"
                f"- FOR CAR: Use 'parked', 'private', 'sedan', 'utility'\n"
                f"  AND PARKING-SPECIFIC: 'parked_vehicle', 'stationary_car', 'private_automobile'\n"
                f"- FOR CLUTTER: Use 'mixed', 'background', 'scattered', 'irregular', 'sparse', 'transition', 'edge'\n"
                f"  AND BOUNDARY/SPECIAL FEATURES: 'fence_line', 'property_boundary', 'demarcation', 'long_shadow', 'tree_shadow'\n"
                f"  AND WINTER-SPECIFIC: 'long_shadow', 'tree_shadow', 'building_shadow', 'small_debris', 'ground_litter', 'unidentified_object'\n\n"
                
                f"AVOID USING:\n"
                f"- For TREE: 'clustered', 'canopy', 'forest', 'wooded' (implies dense vegetation, not sparse trees)\n"
                f"- For BUILDING: 'clustered', 'dense', 'urban' (implies dense urban, not Potsdam's suburban nature)\n"
                f"- For GRASS/LOW_VEG: 'manicured', 'lawn', 'maintained' (implies well-kept, not natural/wild)\n"
                f"- For CAR: 'adjacent', 'connected' (describes relationships, not objects)\n"
                f"- FOR CLUTTER: 'transition', 'boundary' (vague spatial terms)\n\n"
                
                f"CRITICAL TREE DESCRIPTIONS:\n"
                f"- Emphasize 'sparse', 'isolated', 'single', 'small crown' for trees as they are typically standalone\n"
                f"- Avoid 'clustered', 'forest', 'canopy' which apply to dense vegetation areas\n"
                f"- Trees in Potsdam often appear as individual specimens along roadsides or in yards\n"
                f"- SEASONAL CHARACTERISTICS: In winter scenes, emphasize structural features like 'bare tree', 'branch', 'trunk', 'limb'\n"
                f"- Seasonal characteristics: 'dormant', 'leafless', 'winter', 'deciduous'\n"
                f"- Structural features: 'complex', 'entangled', 'interlacing' (for branch patterns)\n"
                f"- Texture features: 'rough', 'bark', 'gnarled'\n"
                f"- Shape features: 'conical', 'oval', 'spreading', 'rounded'\n"
                
                f"CRITICAL CLUTTER DESCRIPTIONS:\n"
                f"- For clutter, use 'background', 'irregular', 'mixed', 'scattered', 'sparse', 'fragmented'\n"
                f"- Clutter represents transitional zones, mixed pixels, and difficult-to-classify areas\n"
                f"- SEASONAL/SHADOW ELEMENTS: Include 'long_shadow', 'tree_shadow', 'building_shadow' as these are common in winter images\n"
                f"- BOUNDARIES: Include 'fence_line', 'property_boundary', 'demarcation' for linear structures\n"
                f"- OTHER ELEMENTS: 'small_debris', 'ground_litter', 'unidentified_object'\n"
                f"- Avoid 'transition' alone (too vague), combine with other terms like 'transition zone'\n"
            )
        elif dataset_type == "vaihingen":
            # Vaihingen dataset context - urban areas
            context_desc = (
                f"VAIHINGEN DATASET CONTEXT (URBAN):\n"
                f"- View perspective: Nadir (top-down) view over urban areas\n"
                f"- Typical characteristics in Vaihingen imagery:\n"
                f"  • Dense urban structures with compact buildings\n"
                f"  • Well-maintained paved surfaces and roads\n"
                f"  • Manicured vegetation in parks and gardens\n"
                f"  • High density of cars in parking lots and streets\n"
                f"  • Regular patterns due to planned urban development\n\n"
                
                f"ADDITIONAL VAIHINGEN CLASS-SPECIFIC HINTS:\n"
                f"- For TREE: 'park', 'garden', 'manicured', 'planned', 'lined', 'ornamental'\n"
                f"- For IMPERVIOUS_SURFACE: 'paved', 'concrete', 'asphalt', 'urban', 'developed'\n"
                f"- FOR LOW_VEGETATION: 'lawn', 'grass', 'manicured', 'park', 'maintained', 'turf'\n"
                f"- FOR BUILDING: 'structure', 'edifice', 'urban', 'dense', 'constructed', 'roofed'\n"
                f"- FOR CAR: 'parked', 'compact', 'row', 'lot', 'organized', 'urban'\n"
                f"- FOR CLUTTER: 'background', 'miscellaneous', 'unstructured', 'mixed', 'varied'\n\n"
            )
        else:
            # Default context for other datasets
            context_desc = (
                f"REMOTE SENSING CONTEXT:\n"
                f"- View perspective: Nadir (top-down) view\n"
                f"- Typical characteristics in aerial imagery:\n"
                f"  • Objects appear from an overhead perspective\n"
                f"  • Colors and textures may vary due to lighting and season\n"
                f"  • Shapes are emphasized due to top-down view\n"
                f"  • Context and arrangement help identification\n\n"
            )

        # Enhanced prompt with general knowledge applicable to various datasets
        prompt = (
            f"<|system|>\nYou are an expert remote sensing image analyst. "
            f"Generate descriptions using ONLY concepts aligned with these classes: {class_list_str}</s>\n"
            f"<|user|>\n<|image_1|>\n"
            f"<|user|>\n<|image_1|>\n"
            f"Analyze this aerial/satellite image for the following land cover and object classes: {class_list_str}\n\n"
            
            f"CRITICAL CONSTRAINTS:\n"
            f"- ALL generated descriptions MUST BE 1-2 WORDS MAXIMUM\n"
            f"- NO PHRASES WITH MORE THAN 2 WORDS\n"
            f"- PREFER SINGLE WORDS OVER TWO-WORD PHRASES\n"
            f"- Examples of ACCEPTABLE: 'building', 'roof', 'car', 'vehicle', 'tree', 'forest'\n"
            f"- Examples of UNACCEPTABLE: 'rectangular building', 'parking lot', 'mixed pixels area'\n\n"
            
            f"CLASSIFICATION RULES:\n"
            f"- GENERAL CLASSES (area-based concepts): {general_classes}\n"
            f"  For these, generate broader semantic synonyms\n"
            f"- INSTANCE CLASSES (specific objects): {list(instance_classes)}\n"
            f"  For these, generate visual/appearance descriptors\n\n"
            f"KNOWLEDGE FOR DIFFERENT CLASS TYPES:\n\n"
            
            f"1. GENERAL OBJECTS (e.g., background, clutter, impervious_surface, low_vegetation):\n"
            f"   - Focus on texture, material, and broad spatial patterns\n"
            f"   - Acceptable prompts: 'natural', 'developed', 'urban', 'rural', 'mixed', 'heterogeneous',\n"
            f"     'paved', 'concrete', 'asphalt', 'grass', 'herbaceous', 'groundcover', 'meadow'\n\n"
            
            f"2. SPECIFIC OBJECTS (e.g., buildings, vehicles, infrastructure):\n"
            f"   - Focus on shape, function, and visual appearance\n"
            f"   - Acceptable prompts: 'structure', 'facility', 'transport', 'vessel', 'craft',\n"
            f"     'container', 'tank', 'cylinder', 'rectangular', 'circular', 'elongated'\n\n"
            f"{context_desc}"
            
            f"GENERAL CLASS EXAMPLES:\n"
            f"- For ROAD: 'street', 'path', 'lane', 'pavement', 'thoroughfare', 'linear'\n"
            f"- FOR BUILDING: 'structure', 'house', 'edifice', 'construction', 'architecture', 'facade'\n"
            f"- FOR TREE: 'forest', 'wood', 'woods', 'grove', 'timber', 'canopy'\n"
            f"- FOR CAR: 'vehicle', 'automobile', 'truck', 'van', 'motorcar', 'transport'\n"
            f"- FOR GRASS: 'lawn', 'meadow', 'pasture', 'turf', 'greensward', 'vegetation'\n\n"
            
            f"For EACH class, provide a structured analysis:\n\n"
            
            f"1. **Instance Count**: How many distinct objects can you identify?\n\n"
            f"2. **Instance-Level Features** (for detecting individual objects):\n"
            f"   - Visual descriptors focusing on shape, texture, and appearance\n\n"
            f"3. **Semantic-Level Context** (for understanding spatial patterns):\n"
            f"   - Spatial descriptors: 'clustered', 'aligned', 'surrounding', 'scattered', 'edge'\n\n"
            f"4. **Detection Confidence**: Your certainty level (high/medium/low)\n\n"
            
            f"FORMAT your response EXACTLY as:\n\n"
            
            f"- {class_names[0]}:\n"
            f"  Count: <number>\n"
            f"  Instance: ['word'], ['word'], ['word']\n"
            f"  Semantic: ['word'], ['word'], ['word']\n"
            f"  Confidence: <high/medium/low>\n\n"
            
            f"- {class_names[1]}:\n"
            f"  Count: <number>\n"
            f"  Instance: ['word'], ['word'], ['word']\n"
            f"  Semantic: ['word'], ['word'], ['word']\n"
            f"  Confidence: <high/medium/low>\n\n"
            
            f"(Continue for ALL classes)\n\n"
            
            f"STRICT GUIDELINES:\n"
            f"- NEVER generate phrases with more than 2 words\n"
            f"- Examples of what NOT to do: 'mixed pixels area', 'rectangular building', 'parking lot'\n"
            f"- Instead of 'parking lot', use 'parked' or 'lot'\n"
            f"- Instead of 'mixed pixels', use 'mixed' or 'pixels'\n"
            f"- Instead of 'linear grid', use 'linear' or 'grid'\n"
            f"- Focus on single-word descriptors!\n"
            f"- CRITICAL: All phrases MUST be 1-2 words, never more!\n"
            f"- ADAPT to the specific image content and dataset characteristics.\n</s>\n"
            
            f"<|assistant|>\n"
        )
        
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response with more focused parameters for complex datasets
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # Increased for detailed hierarchical responses for complex datasets
                do_sample=True,
                temperature=0.6,  # Slightly lower for more consistent outputs
                top_p=0.8,       # Slightly lower for more focused outputs
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.1  # Reduce repetitive outputs
            )
        
        # Clear GPU cache to free memory
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        # Decode the response - only get the newly generated part
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # DEBUG: Print raw Qwen3 response to diagnose the issue
        print(f"\n[DEBUG] Raw Qwen3 Response for {dataset_type} dataset:")
        print("=" * 80)
        print(response[:2000])  # Print first 2000 chars
        print("=" * 80)
        
        # Parse the response to extract hierarchical information
        result = {}
        
        print(f"\nParsing LLM response for {len(class_names)} classes on {dataset_type} dataset:")
        print("=" * 60)
        
        for class_name in class_names:
            try:
                # Extract the block for this class
                class_pattern = rf"{re.escape(class_name)}\s*:(.*?)(?=\n\s*- \w+:|\Z)"
                class_match = re.search(class_pattern, response, re.DOTALL | re.IGNORECASE)
                
                if not class_match:
                    # Fallback if pattern not found
                    print(f"  ⚠ Warning: Could not parse response for '{class_name}'")
                    result[class_name] = {
                        'count': 0,
                        'descriptions': [class_name],
                        'instance_features': [class_name],
                        'semantic_features': [],
                        'confidence': 'low'
                    }
                    continue
                
                class_block = class_match.group(1)
                
                # Extract count
                count_match = re.search(r"Count\s*:\s*(\d+)", class_block, re.IGNORECASE)
                count = int(count_match.group(1)) if count_match else 0
                
                # Extract instance features - handle bracket format
                instance_match = re.search(r"Instance\s*:\s*\[(.*?)\]", class_block, re.IGNORECASE | re.DOTALL)
                instance_features = []
                if instance_match:
                    instance_text = instance_match.group(1)
                    # Split by ],[ or comma
                    raw_features = re.split(r'\]\s*,\s*\[|,', instance_text)
                    # Clean and filter features with specialized handling for challenging classes
                    for f in raw_features:
                        f = f.strip().strip('"\'').strip("'")
                        # Split by spaces and take only first 1-2 words
                        words = f.split()
                        if len(words) == 1:
                            # Single word - accept as is
                            processed_word = words[0].lower()
                            if processed_word and processed_word not in instance_features:
                                # Additional check to ensure relevance to the class
                                instance_features.append(processed_word)
                        elif len(words) == 2:
                            # Two words - accept if both are meaningful
                            two_word_phrase = ' '.join(words).lower()
                            if all(w.isalpha() for w in words) and two_word_phrase not in instance_features:
                                instance_features.append(two_word_phrase)
                        # We'll keep the original logic and avoid overly specific terms
                        
                # Apply dataset-specific constraints to filter inappropriate terms
                if dataset_type == "potsdam":
                    filtered_instance_features = []
                    for feature in instance_features:
                        # Skip inappropriate terms for Potsdam dataset
                        if class_name in ['tree', 'vegetation'] and feature in ['clustered', 'canopy', 'forest', 'wooded', 'aligned']:
                            continue
                        elif class_name in ['building', 'structure'] and feature in ['clustered', 'dense', 'urban', 'surrounding', 'adjacent', 'connected']:
                            continue
                        elif class_name in ['low_vegetation', 'grass'] and feature in ['manicured', 'lawn', 'maintained', 'park', 'garden']:
                            continue
                        elif class_name in ['car', 'vehicle'] and feature in ['adjacent', 'connected', 'aligned']:
                            continue
                        elif class_name in ['road', 'path'] and feature in ['aligned', 'connected']:
                            continue
                        else:
                            filtered_instance_features.append(feature)
                    instance_features = filtered_instance_features
                
                # If still no instance features after filtering, use the class name itself as fallback
                if not instance_features:
                    # Provide dataset-specific fallbacks that work across different datasets
                    if dataset_type == "potsdam":
                        if class_name in ['tree', 'vegetation']:
                            # 根据Potsdam数据集特点，特别强调孤立树的特征
                            instance_features = ['sparse', 'isolated', 'single', 'small', 'individual', 'standalone', 'bare_tree', 'branch', 'trunk', 'limb', 'dormant', 'leafless', 'winter', 'deciduous', 'complex', 'entangled', 'interlacing', 'rough', 'bark', 'gnarled', 'conical', 'oval', 'spreading', 'rounded', 'dormant_tree', 'winter_tree', 'leafless_tree', 'branch_structure', 'tree_skeleton', 'isolated_trunk', 'single_tree', 'scattered_tree', 'winter_bare', 'leafless_branch', 'bare_branch', 'tree_trunk', 'wooden_stem']
                        elif class_name in ['low_vegetation', 'grass']:
                            instance_features = ['sparse', 'natural', 'overgrown', 'wild', 'grass', 'patchy', 'dormant_grass', 'winter_ground_cover', 'browned_vegetation', 'dry_grassland', 'barren_field', 'winter_vegetation', 'dormant_vegetation', 'brown_grass', 'dry_vegetation']
                        elif class_name in ['building', 'structure']:
                            # 根据Potsdam数据集特点，强调独立住宅特征，避免使用'clustered'、'dense'等词汇
                            instance_features = ['house', 'home', 'cottage', 'residential', 'dwelling', 'garden', 'yard', 'private', 'suburban', 'detached', 'roof', 'gable', 'chimney', 'facade', 'brick', 'tile', 'residential_roof', 'house_structure', 'rural_dwelling', 'detached_house', 'suburban_home', 'garden_house']
                        elif class_name in ['impervious_surface', 'road']:
                            instance_features = ['dirt', 'gravel', 'stone', 'unpaved', 'rough', 'soil', 'dirt_road', 'earthen_path', 'unpaved_trail', 'rural_pathway', 'country_track', 'field_path', 'unpaved_surface', 'soil_surface', 'gravel_surface', 'dirt_surface']
                        elif class_name in ['car', 'vehicle']:
                            instance_features = ['parked', 'private', 'sedan', 'utility', 'personal', 'parked_vehicle', 'stationary_car', 'private_automobile', 'parked_auto', 'idle_vehicle', 'parked_transport']
                        elif class_name in ['background', 'clutter']:
                            instance_features = ['background', 'irregular', 'mixed', 'scattered', 'sparse', 'fragmented', 'fence_line', 'property_boundary', 'demarcation', 'long_shadow', 'tree_shadow', 'building_shadow', 'small_debris', 'ground_litter', 'unidentified_object', 'shadow_area', 'fence_post', 'linear_structure', 'boundary_feature', 'structure_edge', 'transition_zone', 'mixed_pixels']
                        elif class_name in ['water', 'river', 'lake']:
                            instance_features = ['pond', 'puddle', 'reflective', 'liquid', 'wet', 'shiny']
                        elif class_name in ['harbor', 'port']:
                            instance_features = ['dock', 'marina', 'waterfront', 'nautical', 'berth', 'quay']
                        elif class_name in ['bridge', 'crossing']:
                            instance_features = ['span', 'connection', 'link', 'crossing', 'structure', 'arch']
                        elif class_name in ['ship', 'vessel']:
                            instance_features = ['marine', 'nautical', 'watercraft', 'floating', 'vessel', 'boat']
                        elif class_name in ['plane', 'aircraft']:
                            instance_features = ['aviation', 'aerial', 'wing', 'flight', 'aircraft', 'hangar']
                        elif class_name in ['field', 'ground']:
                            instance_features = ['terrain', 'surface', 'area', 'tract', 'zone', 'patch']
                        elif class_name in ['tank', 'container']:
                            instance_features = ['storage', 'vessel', 'cylinder', 'reservoir', 'facility', 'unit']
                        else:
                            # Generic fallback for any class
                            instance_features = [class_name, f'{class_name}-object', f'{class_name}-item', 
                                                 'object', 'entity', 'thing']
                    elif dataset_type == "vaihingen":
                        if class_name in ['background', 'clutter']:
                            instance_features = ['background', 'clutter', 'miscellaneous', 'unstructured', 'mixed', 'varied']
                        elif class_name in ['building', 'structure']:
                            instance_features = ['structure', 'edifice', 'urban', 'dense', 'constructed', 'roofed']
                        elif class_name in ['tree', 'vegetation']:
                            instance_features = ['park', 'garden', 'manicured', 'planned', 'lined', 'ornamental']
                        elif class_name in ['road', 'path']:
                            instance_features = ['paved', 'concrete', 'asphalt', 'urban', 'developed', 'road']
                        elif class_name in ['car', 'vehicle']:
                            instance_features = ['parked', 'compact', 'row', 'lot', 'organized', 'urban']
                        else:
                            # Generic fallback for any class
                            instance_features = [class_name, f'{class_name}-object', f'{class_name}-item', 
                                                 'object', 'entity', 'thing']
                    else:
                        # Default fallback for other datasets
                        if class_name in ['background', 'clutter']:
                            instance_features = ['scattered', 'irregular', 'mixed', 'boundary', 'edge', 'transition']
                        elif class_name in ['building', 'structure']:
                            instance_features = ['house', 'home', 'cottage', 'residential', 'dwelling', 'garden']
                        elif class_name in ['tree', 'vegetation']:
                            instance_features = ['isolated', 'standalone', 'sparse', 'natural', 'wild', 'single']
                        elif class_name in ['road', 'path']:
                            instance_features = ['dirt', 'gravel', 'stone', 'unpaved', 'rough', 'soil']
                        elif class_name in ['car', 'vehicle']:
                            instance_features = ['parked', 'private', 'sedan', 'utility', 'personal', 'small']
                        elif class_name in ['water', 'river', 'lake']:
                            instance_features = ['pond', 'puddle', 'reflective', 'liquid', 'wet', 'shiny']
                        elif class_name in ['harbor', 'port']:
                            instance_features = ['dock', 'marina', 'waterfront', 'nautical', 'berth', 'quay']
                        elif class_name in ['bridge', 'crossing']:
                            instance_features = ['span', 'connection', 'link', 'crossing', 'structure', 'arch']
                        elif class_name in ['ship', 'vessel']:
                            instance_features = ['marine', 'nautical', 'watercraft', 'floating', 'vessel', 'boat']
                        elif class_name in ['plane', 'aircraft']:
                            instance_features = ['aviation', 'aerial', 'wing', 'flight', 'aircraft', 'hangar']
                        elif class_name in ['field', 'ground']:
                            instance_features = ['terrain', 'surface', 'area', 'tract', 'zone', 'patch']
                        elif class_name in ['tank', 'container']:
                            instance_features = ['storage', 'vessel', 'cylinder', 'reservoir', 'facility', 'unit']
                        else:
                            # Generic fallback for any class
                            instance_features = [class_name, f'{class_name}-object', f'{class_name}-item', 
                                                 'object', 'entity', 'thing']
                
                # Extract semantic features - handle bracket format
                semantic_match = re.search(r"Semantic\s*:\s*\[(.*?)\]", class_block, re.IGNORECASE | re.DOTALL)
                semantic_features = []
                if semantic_match:
                    semantic_text = semantic_match.group(1)
                    # Split by ],[ or comma
                    raw_features = re.split(r'\]\s*,\s*\[|,', semantic_text)
                    # Clean and filter features
                    for f in raw_features:
                        f = f.strip().strip('"\'').strip("'")
                        # Split by spaces and take only first 1-2 words
                        words = f.split()
                        if len(words) == 1:
                            # Single word - accept as is
                            processed_word = words[0].lower()
                            if processed_word and processed_word not in semantic_features:
                                semantic_features.append(processed_word)
                        elif len(words) == 2:
                            # Two words - accept if both are meaningful
                            two_word_phrase = ' '.join(words).lower()
                            if all(w.isalpha() for w in words) and two_word_phrase not in semantic_features:
                                semantic_features.append(two_word_phrase)
                        # We'll keep the original logic and avoid overly specific terms
                    
                    # Apply dataset-specific constraints to filter inappropriate terms for semantic features too
                    if dataset_type == "potsdam":
                        filtered_semantic_features = []
                        for feature in semantic_features:
                            # Skip inappropriate terms for Potsdam dataset
                            if class_name in ['tree', 'vegetation'] and feature in ['clustered', 'canopy', 'forest', 'wooded', 'aligned']:
                                continue
                            elif class_name in ['building', 'structure'] and feature in ['clustered', 'dense', 'urban', 'surrounding', 'adjacent', 'connected']:
                                continue
                            elif class_name in ['low_vegetation', 'grass'] and feature in ['manicured', 'lawn', 'maintained', 'park', 'garden']:
                                continue
                            elif class_name in ['car', 'vehicle'] and feature in ['adjacent', 'connected', 'aligned']:
                                continue
                            elif class_name in ['road', 'path'] and feature in ['aligned', 'connected']:
                                continue
                            else:
                                filtered_semantic_features.append(feature)
                        semantic_features = filtered_semantic_features
                
                # If no semantic features extracted, provide dataset-specific defaults
                if not semantic_features:
                    if dataset_type == "potsdam":
                        if class_name in ['tree', 'vegetation']:
                            semantic_features = ['sparse', 'isolated', 'scattered', 'individual', 'standalone', 'small', 'winter_tree', 'leafless_tree', 'dormant_tree', 'branch_structure', 'tree_skeleton']
                        elif class_name in ['low_vegetation', 'grass']:
                            semantic_features = ['sparse', 'natural', 'wild', 'overgrown', 'patchy', 'uneven', 'dormant_grass', 'winter_ground_cover', 'browned_vegetation']
                        elif class_name in ['building', 'structure']:
                            semantic_features = ['suburban', 'residential', 'garden', 'yard', 'home', 'private', 'rural_dwelling']
                        elif class_name in ['impervious_surface', 'road']:
                            semantic_features = ['rural', 'dirt', 'unpaved', 'gravel', 'rough', 'country', 'dirt_road', 'earthen_path', 'unpaved_trail', 'rural_pathway']
                        elif class_name in ['car', 'vehicle']:
                            semantic_features = ['private', 'parked', 'residential', 'driveway', 'garage', 'sedan', 'parked_vehicle', 'stationary_car']
                        elif class_name in ['background', 'clutter']:
                            semantic_features = ['background', 'irregular', 'scattered', 'sparse', 'fragmented', 'mixed', 'fence_line', 'property_boundary', 'demarcation', 'long_shadow', 'tree_shadow', 'building_shadow', 'small_debris', 'ground_litter']
                        elif class_name in ['water', 'river', 'lake']:
                            semantic_features = ['pond', 'puddle', 'reflective', 'wetland', 'moist', 'shiny']
                        elif class_name in ['harbor', 'port']:
                            semantic_features = ['maritime', 'nautical', 'coastal', 'shipping', 'waterfront', 'berth']
                        elif class_name in ['bridge', 'crossing']:
                            semantic_features = ['infrastructure', 'connection', 'span', 'engineering', 'link', 'crossing']
                        elif class_name in ['ship', 'vessel']:
                            semantic_features = ['maritime', 'nautical', 'floating', 'vessel', 'naval', 'watercraft']
                        elif class_name in ['plane', 'aircraft']:
                            semantic_features = ['aviation', 'aerial', 'flight', 'airborne', 'winged', 'airport']
                        elif class_name in ['field', 'ground']:
                            semantic_features = ['terrain', 'landscape', 'area', 'tract', 'zone', 'patch']
                        elif class_name in ['tank', 'container']:
                            semantic_features = ['storage', 'industrial', 'facility', 'container', 'reservoir', 'unit']
                        else:
                            # Generic fallback for any class
                            semantic_features = ['spatial', 'contextual', 'environmental', 'functional', 'relational', 'adjacent', 'surrounding']
                    elif dataset_type == "vaihingen":
                        if class_name in ['background', 'clutter']:
                            semantic_features = ['context', 'environment', 'surroundings', 'setting', 'space', 'urban', 'city']
                        elif class_name in ['building', 'structure']:
                            semantic_features = ['urban', 'development', 'construction', 'architecture', 'edifice', 'dense', 'city']
                        elif class_name in ['tree', 'vegetation']:
                            semantic_features = ['natural', 'environmental', 'green', 'floral', 'ecological', 'park', 'garden']
                        elif class_name in ['road', 'path']:
                            semantic_features = ['transport', 'connectivity', 'network', 'infrastructure', 'route', 'urban', 'paved']
                        elif class_name in ['car', 'vehicle']:
                            semantic_features = ['mobility', 'transportation', 'mechanical', 'automotive', 'movement', 'parked', 'city']
                        else:
                            # Generic fallback for any class
                            semantic_features = ['spatial', 'contextual', 'environmental', 'functional', 'relational', 'urban', 'developed']
                    else:
                        # Default for other datasets
                        if class_name in ['background', 'clutter']:
                            semantic_features = ['scattered', 'irregular', 'mixed', 'boundary', 'edge', 'transition']
                        elif class_name in ['building', 'structure']:
                            semantic_features = ['suburban', 'residential', 'garden', 'yard', 'home', 'private', 'dwelling']
                        elif class_name in ['tree', 'vegetation']:
                            semantic_features = ['natural', 'wild', 'sparse', 'scattered', 'overgrown', 'bushy', 'leafy']
                        elif class_name in ['road', 'path']:
                            semantic_features = ['rural', 'dirt', 'unpaved', 'gravel', 'rough', 'country', 'access']
                        elif class_name in ['car', 'vehicle']:
                            semantic_features = ['private', 'parked', 'residential', 'driveway', 'garage', 'sedan', 'utility']
                        elif class_name in ['water', 'river', 'lake']:
                            semantic_features = ['pond', 'puddle', 'reflective', 'wetland', 'moist', 'shiny', 'glassy']
                        elif class_name in ['harbor', 'port']:
                            semantic_features = ['maritime', 'nautical', 'coastal', 'shipping', 'waterfront', 'berth', 'dock']
                        elif class_name in ['bridge', 'crossing']:
                            semantic_features = ['infrastructure', 'connection', 'span', 'engineering', 'link', 'crossing', 'passage']
                        elif class_name in ['ship', 'vessel']:
                            semantic_features = ['maritime', 'nautical', 'floating', 'vessel', 'naval', 'watercraft', 'boat']
                        elif class_name in ['plane', 'aircraft']:
                            semantic_features = ['aviation', 'aerial', 'flight', 'airborne', 'winged', 'airport', 'hangar']
                        elif class_name in ['field', 'ground']:
                            semantic_features = ['terrain', 'landscape', 'area', 'tract', 'zone', 'patch', 'plot']
                        elif class_name in ['tank', 'container']:
                            semantic_features = ['storage', 'industrial', 'facility', 'container', 'reservoir', 'unit', 'depot']
                        else:
                            # Generic fallback for any class
                            semantic_features = ['spatial', 'contextual', 'environmental', 'functional', 'relational']
                
                # Extract confidence
                conf_match = re.search(r"Confidence\s*:\s*(high|medium|low)", class_block, re.IGNORECASE)
                confidence = conf_match.group(1).lower() if conf_match else 'low'
                
                # Build description list by combining instance and semantic features
                descriptions = [class_name]  # Always include baseline
                
                # Add instance features (prioritize for SAM3 instance head)
                for feature in instance_features[:3]:
                    if feature and feature not in descriptions:
                        descriptions.append(feature)
                
                # Add semantic features if space allows
                remaining_slots = 5 - len(descriptions)
                if remaining_slots > 0:
                    for feature in semantic_features[:remaining_slots]:
                        if feature and feature not in descriptions:
                            descriptions.append(feature)
                
                result[class_name] = {
                    'count': count,
                    'descriptions': descriptions,
                    'instance_features': instance_features,
                    'semantic_features': semantic_features,
                    'confidence': confidence
                }
                
                # Print parsed features for debugging with remote sensing context
                print(f"  ✓ {class_name}: {descriptions}")
                print(f"    Count: {count}, Confidence: {confidence}")
                if instance_features:
                    print(f"    Instance (top-down view): {instance_features}")
                if semantic_features:
                    print(f"    Semantic (spatial pattern): {semantic_features}")
                
            except Exception as e:
                print(f"  ⚠ Warning: Failed to parse response for class '{class_name}': {e}")
                # Fallback on error
                result[class_name] = {
                    'count': 0,
                    'descriptions': [class_name],
                    'instance_features': [class_name],
                    'semantic_features': [],
                    'confidence': 'low'
                }
        
        return result

    def generate_scene_description(self, image_path: str) -> str:
        """
        Generate a scene description for the given image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Description of the scene
        """
        # Create a prompt with image for scene description
        prompt = (
            f"<|system|>\nYou are a helpful assistant.</s>\n"
            f"<|user|>\n<|image_1|>\nDescribe the scene in this remote sensing image. Focus on the main objects and land cover types present.</s>\n"
            f"<|assistant|>\n"
        )
        
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,  # Further reduced for memory
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,  # Prevent warning
                use_cache=True,
            )
        
        # Clear GPU cache to free memory
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        # Decode the response - only get the newly generated part
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return response if response else f"Scene description for {image_path} - Placeholder"