from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import custom_datasets  # noqa: F401
import sam3_segmentor_cached
from mmengine.config import Config


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_DATA_ROOT = REPO_ROOT / "QwSAM3TestData"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "expanded_prompt_aggregation_visualizations"
SUPPORTED_DATASET_CONFIGS = {
    "iSAID": REPO_ROOT / "configs" / "cfg_iSAID_controlled.py",
    "LoveDA": REPO_ROOT / "configs" / "cfg_loveda_controlled.py",
    "Potsdam": REPO_ROOT / "configs" / "cfg_potsdam_controlled.py",
    "Vaihingen": REPO_ROOT / "configs" / "cfg_vaihingen_controlled.py",
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize how expanded prompts produce semantic/instance/fused maps "
            "and how they are aggregated into the final class logit."
        )
    )
    parser.add_argument(
        "--test-data-root",
        default=str(DEFAULT_TEST_DATA_ROOT),
        help="Root directory containing per-dataset sample images.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used to save the generated figures.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional subset of dataset folder names under QwSAM3TestData.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=3,
        help="Number of random images to visualize per dataset.",
    )
    parser.add_argument(
        "--max-expanded-prompts",
        type=int,
        default=4,
        help="Maximum number of expanded prompts to visualize for the selected class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for image sampling.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for model inference.",
    )
    return parser.parse_args()


def normalize_optimize_method(optimize_method: str) -> str:
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
            return canonical_name
    return str(optimize_method)


def resolve_expanded_prompt_pool_path(config_path: Path, cfg: Config) -> str | None:
    model_cfg = cfg.get("model", {})
    prompt_pool_path = model_cfg.get("expanded_prompt_pool_path")
    if prompt_pool_path:
        return prompt_pool_path
    if not model_cfg.get("enable_expanded_prompt", False):
        return None
    optimize_method = normalize_optimize_method(model_cfg.get("optimize_method", "guided"))
    return f"./prompt_pools/{config_path.stem}_expanded_prompt_pool_{optimize_method}.pkl"


def load_base_class_names(classname_path: str) -> list[str]:
    query_words, query_idx = sam3_segmentor_cached.get_cls_idx(classname_path)
    class_names: list[str] = []
    seen = set()
    for word, idx in zip(query_words, query_idx):
        if idx in seen:
            continue
        seen.add(idx)
        class_names.append(word)
    return class_names


def build_model_from_config(
    config_path: Path,
    device: str,
    *,
    enable_expanded_prompt: bool,
    expanded_prompt_pool_path: str | None,
):
    cfg = Config.fromfile(str(config_path))
    model_cfg = dict(cfg.model)
    model_cfg.pop("type", None)
    model_cfg["device"] = device
    model_cfg["enable_expanded_prompt"] = enable_expanded_prompt
    model_cfg["expanded_prompt_pool_path"] = expanded_prompt_pool_path
    model = sam3_segmentor_cached.CachedSAM3OpenSegmentor(**model_cfg)
    model.eval()
    return model, cfg


def sample_dataset_images(dataset_dir: Path, sample_count: int, rng: random.Random) -> list[Path]:
    image_paths = sorted(
        path for path in dataset_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if sample_count >= len(image_paths):
        return image_paths
    return sorted(rng.sample(image_paths, sample_count))


def run_single_prompt_raw_outputs(model, pil_image: Image.Image, prompt_idx: int):
    engine = model.execution_engine
    with torch.no_grad():
        with engine._encoder_autocast_context():
            inference_state = engine.processor.set_image(pil_image)
        backbone_out_image_only = inference_state["backbone_out"]
        backbone_out_image_only = engine._cast_backbone_out_to_decoder_dtype(
            backbone_out_image_only
        )
        engine._prepare_cached_image_features(backbone_out_image_only)

        query_indices = torch.tensor([prompt_idx], device=engine.device, dtype=torch.long)
        backbone_out = model.prompt_bank.build_backbone_out_by_indices(
            backbone_out_image_only=backbone_out_image_only,
            query_indices=query_indices,
            image_repeats=1,
        )
        find_stage, dummy_prompt = engine._get_prompt_batch_context(1, 1)
        with engine._decoder_autocast_context():
            outputs = engine.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_stage,
                geometric_prompt=dummy_prompt,
                find_target=None,
            )
    return outputs


def compute_prompt_fusion_artifacts(outputs, output_size: tuple[int, int], confidence_threshold: float):
    height, width = output_size

    semantic_logits = outputs["semantic_seg"]
    if semantic_logits.ndim == 3:
        semantic_logits = semantic_logits.unsqueeze(1)
    if semantic_logits.shape[-2:] != (height, width):
        semantic_logits = torch.nn.functional.interpolate(
            semantic_logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    semantic_hw = semantic_logits[0, 0].sigmoid().detach().cpu()

    pred_masks = outputs["pred_masks"]
    if pred_masks.ndim == 5 and pred_masks.shape[2] == 1:
        pred_masks = pred_masks.squeeze(2)
    if pred_masks.shape[-2:] != (height, width):
        pred_masks = torch.nn.functional.interpolate(
            pred_masks.reshape(-1, 1, *pred_masks.shape[-2:]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(1, pred_masks.shape[1], height, width)

    pred_scores = outputs["pred_logits"].sigmoid()
    if pred_scores.ndim == 3 and pred_scores.shape[-1] == 1:
        pred_scores = pred_scores.squeeze(-1)
    pred_scores = pred_scores[0]
    valid_scores = pred_scores > confidence_threshold

    instance_probs_nhw = pred_masks[0].sigmoid().detach().cpu()
    weighted_instance_nhw = (
        pred_masks[0].sigmoid() * (pred_scores * valid_scores).view(-1, 1, 1)
    ).detach().cpu()
    instance_max_hw = weighted_instance_nhw.amax(dim=0)
    fused_hw = torch.maximum(semantic_hw, instance_max_hw)
    source_map = (instance_max_hw > semantic_hw).numpy().astype(np.int64)

    return {
        "semantic_hw": semantic_hw.numpy(),
        "instance_max_hw": instance_max_hw.numpy(),
        "fused_hw": fused_hw.numpy(),
        "source_map": source_map,
        "pred_scores": pred_scores.detach().cpu().numpy(),
    }


def aggregate_query_logits_by_class(model, query_logits: torch.Tensor) -> torch.Tensor:
    if query_logits.shape[0] == model.num_cls:
        return query_logits.detach().cpu()
    query_idx = model.query_idx.to(device=query_logits.device, dtype=torch.long)
    cls_index = torch.nn.functional.one_hot(
        query_idx, num_classes=model.num_cls
    ).T.view(model.num_cls, len(model.query_idx), 1, 1)
    class_logits = (query_logits.unsqueeze(0) * cls_index).max(1)[0]
    return class_logits.detach().cpu()


def select_target_class(base_model, expanded_model, base_class_logits: torch.Tensor, max_prompts: int) -> int:
    expanded_prompt_counts = Counter(int(idx.item()) for idx in expanded_model.query_idx)
    base_prompt_counts = Counter(int(idx.item()) for idx in base_model.query_idx)
    class_scores = base_class_logits.flatten(1).amax(dim=1)
    ranked_classes = torch.argsort(class_scores, descending=True).tolist()
    for cls_idx in ranked_classes:
        if cls_idx == int(base_model.bg_idx):
            continue
        if expanded_prompt_counts.get(int(cls_idx), 0) > base_prompt_counts.get(int(cls_idx), 0):
            return int(cls_idx)
    for cls_idx in ranked_classes:
        if cls_idx != int(base_model.bg_idx):
            return int(cls_idx)
    return int(ranked_classes[0])


def build_palette(num_classes: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20", max(num_classes, 1))
    palette = np.array(
        [np.array(cmap(i)[:3]) * 255 for i in range(num_classes)],
        dtype=np.uint8,
    )
    return palette


def overlay_segmentation(image_np: np.ndarray, pred_mask: np.ndarray, palette: np.ndarray, alpha: float = 0.45):
    color_mask = palette[pred_mask]
    blended = image_np.astype(np.float32) * (1.0 - alpha) + color_mask.astype(np.float32) * alpha
    return blended.astype(np.uint8)


def add_text_panel(ax, lines: list[str], title: str | None = None):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=9.5,
        family="monospace",
        transform=ax.transAxes,
    )


def build_expanded_prompt_aggregation_artifacts(
    expanded_model,
    expanded_prompt_indices: list[int],
    expanded_prompt_fused_maps: list[np.ndarray],
):
    prompt_tensor = torch.tensor(
        np.stack(expanded_prompt_fused_maps, axis=0),
        dtype=torch.float32,
    )
    query_indices = expanded_model.query_idx.detach().cpu().index_select(
        0,
        torch.tensor(expanded_prompt_indices, dtype=torch.long),
    )
    class_logits = expanded_model._aggregate_query_logits_to_class_logits(
        prompt_tensor,
        query_indices=query_indices,
    ).detach().cpu()
    target_cls_idx = int(query_indices[0].item())
    target_class_logit = class_logits[target_cls_idx].numpy()

    fused_prompt_tensor = prompt_tensor.numpy()
    winner_map = fused_prompt_tensor.argmax(axis=0).astype(np.int64)
    top2 = np.partition(fused_prompt_tensor, kth=max(fused_prompt_tensor.shape[0] - 2, 0), axis=0)
    sorted_top2 = np.sort(fused_prompt_tensor, axis=0)
    if fused_prompt_tensor.shape[0] > 1:
        margin_map = sorted_top2[-1] - sorted_top2[-2]
    else:
        margin_map = sorted_top2[-1]

    return {
        "aggregated_class_logit": target_class_logit,
        "winner_map": winner_map,
        "margin_map": margin_map,
    }


def visualize_single_image(
    dataset_name: str,
    image_path: Path,
    base_model,
    expanded_model,
    class_names: list[str],
    output_dir: Path,
    max_expanded_prompts: int,
):
    with Image.open(image_path) as image_handle:
        pil_image = image_handle.convert("RGB")
    image_np = np.array(pil_image)

    with torch.no_grad():
        base_query_logits = base_model._inference_single_view(pil_image).detach().cpu()
        expanded_query_logits = expanded_model._inference_single_view(pil_image).detach().cpu()

    base_class_logits = aggregate_query_logits_by_class(base_model, base_query_logits)
    expanded_class_logits = aggregate_query_logits_by_class(expanded_model, expanded_query_logits)
    base_pred = base_class_logits.argmax(dim=0).numpy().astype(np.int64)
    expanded_pred = expanded_class_logits.argmax(dim=0).numpy().astype(np.int64)

    target_cls_idx = select_target_class(
        base_model,
        expanded_model,
        base_class_logits,
        max_prompts=max_expanded_prompts,
    )
    class_name = class_names[target_cls_idx]

    base_prompt_indices = torch.nonzero(
        base_model.query_idx == target_cls_idx, as_tuple=False
    ).flatten().tolist()
    expanded_prompt_indices = torch.nonzero(
        expanded_model.query_idx == target_cls_idx, as_tuple=False
    ).flatten().tolist()

    base_prompt_idx = base_prompt_indices[0]
    base_prompt_word = base_model.query_words[base_prompt_idx]

    expanded_prompt_indices = expanded_prompt_indices[:max_expanded_prompts]
    expanded_prompt_words = [expanded_model.query_words[idx] for idx in expanded_prompt_indices]

    base_outputs = run_single_prompt_raw_outputs(base_model, pil_image, base_prompt_idx)
    base_artifacts = compute_prompt_fusion_artifacts(
        base_outputs,
        output_size=(pil_image.height, pil_image.width),
        confidence_threshold=base_model.confidence_threshold,
    )

    expanded_prompt_artifacts = []
    for prompt_idx, prompt_word in zip(expanded_prompt_indices, expanded_prompt_words):
        prompt_outputs = run_single_prompt_raw_outputs(expanded_model, pil_image, prompt_idx)
        prompt_artifacts = compute_prompt_fusion_artifacts(
            prompt_outputs,
            output_size=(pil_image.height, pil_image.width),
            confidence_threshold=expanded_model.confidence_threshold,
        )
        expanded_prompt_artifacts.append((prompt_idx, prompt_word, prompt_artifacts))

    expanded_aggregation = build_expanded_prompt_aggregation_artifacts(
        expanded_model,
        expanded_prompt_indices,
        [item[2]["fused_hw"] for item in expanded_prompt_artifacts],
    )

    palette = build_palette(len(class_names))
    base_overlay = overlay_segmentation(image_np, base_pred, palette)
    expanded_overlay = overlay_segmentation(image_np, expanded_pred, palette)
    target_change = np.zeros((*base_pred.shape, 3), dtype=np.float32)
    target_change[..., :] = np.array([0.12, 0.12, 0.12], dtype=np.float32)
    lost_mask = (base_pred == target_cls_idx) & (expanded_pred != target_cls_idx)
    gain_mask = (base_pred != target_cls_idx) & (expanded_pred == target_cls_idx)
    target_change[lost_mask] = np.array([0.92, 0.25, 0.25], dtype=np.float32)
    target_change[gain_mask] = np.array([0.25, 0.85, 0.25], dtype=np.float32)

    nrows = 2 + len(expanded_prompt_artifacts)
    ncols = 5
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.3 * nrows),
        squeeze=False,
    )

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(base_overlay)
    axes[0, 1].set_title("Base-Only Segmentation", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(expanded_overlay)
    axes[0, 2].set_title("Expanded Segmentation", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(target_change)
    axes[0, 3].set_title("Target Change Map", fontsize=12, fontweight="bold")
    axes[0, 3].axis("off")

    summary_lines = [
        f"Dataset: {dataset_name}",
        f"Image:   {image_path.name}",
        f"Target class: {class_name}",
        f"Base prompt(s): {len(base_prompt_indices)}",
        f"Expanded prompt(s): {len(expanded_prompt_indices)}",
        "",
        f"Base target score: {float(base_class_logits[target_cls_idx].max().item()):.3f}",
        f"Expanded target score: {float(expanded_class_logits[target_cls_idx].max().item()):.3f}",
        f"Base target pixels: {int((base_pred == target_cls_idx).sum())}",
        f"Expanded target pixels: {int((expanded_pred == target_cls_idx).sum())}",
        f"Lost pixels: {int(lost_mask.sum())}",
        f"Gained pixels: {int(gain_mask.sum())}",
    ]
    add_text_panel(axes[0, 4], summary_lines, title="Workflow Summary")

    base_sem_im = axes[1, 0].imshow(base_artifacts["semantic_hw"], cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title(f"Base Semantic\n{base_prompt_word}", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    fig.colorbar(base_sem_im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    base_inst_im = axes[1, 1].imshow(base_artifacts["instance_max_hw"], cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title(f"Base Instance Max\n{base_prompt_word}", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")
    fig.colorbar(base_inst_im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    base_fused_im = axes[1, 2].imshow(base_artifacts["fused_hw"], cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title(f"Base Fused\n{base_prompt_word}", fontsize=12, fontweight="bold")
    axes[1, 2].axis("off")
    fig.colorbar(base_fused_im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    base_source_im = axes[1, 3].imshow(base_artifacts["source_map"], cmap="coolwarm", vmin=0, vmax=1)
    axes[1, 3].set_title("Base Pixel Source\n0=sem,1=inst", fontsize=12, fontweight="bold")
    axes[1, 3].axis("off")
    fig.colorbar(base_source_im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    base_summary = [
        f"semantic max={float(base_artifacts['semantic_hw'].max()):.3f}",
        f"inst max={float(base_artifacts['instance_max_hw'].max()):.3f}",
        f"fused max={float(base_artifacts['fused_hw'].max()):.3f}",
        f"valid inst={int((base_artifacts['pred_scores'] > base_model.confidence_threshold).sum())}",
    ]
    add_text_panel(axes[1, 4], base_summary, title="Base Prompt Workflow")

    for row_idx, (_, prompt_word, artifacts) in enumerate(expanded_prompt_artifacts, start=2):
        sem_im = axes[row_idx, 0].imshow(artifacts["semantic_hw"], cmap="inferno", vmin=0.0, vmax=1.0)
        axes[row_idx, 0].set_title(f"Semantic\n{prompt_word}", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")
        fig.colorbar(sem_im, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        inst_im = axes[row_idx, 1].imshow(artifacts["instance_max_hw"], cmap="inferno", vmin=0.0, vmax=1.0)
        axes[row_idx, 1].set_title(f"Instance Max\n{prompt_word}", fontsize=11, fontweight="bold")
        axes[row_idx, 1].axis("off")
        fig.colorbar(inst_im, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        fused_im = axes[row_idx, 2].imshow(artifacts["fused_hw"], cmap="inferno", vmin=0.0, vmax=1.0)
        axes[row_idx, 2].set_title(f"Fused\n{prompt_word}", fontsize=11, fontweight="bold")
        axes[row_idx, 2].axis("off")
        fig.colorbar(fused_im, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        source_im = axes[row_idx, 3].imshow(artifacts["source_map"], cmap="coolwarm", vmin=0, vmax=1)
        axes[row_idx, 3].set_title("Pixel Source\n0=sem,1=inst", fontsize=11, fontweight="bold")
        axes[row_idx, 3].axis("off")
        fig.colorbar(source_im, ax=axes[row_idx, 3], fraction=0.046, pad=0.04)

        prompt_summary = [
            f"semantic max={float(artifacts['semantic_hw'].max()):.3f}",
            f"inst max={float(artifacts['instance_max_hw'].max()):.3f}",
            f"fused max={float(artifacts['fused_hw'].max()):.3f}",
            f"valid inst={int((artifacts['pred_scores'] > expanded_model.confidence_threshold).sum())}",
        ]
        add_text_panel(axes[row_idx, 4], prompt_summary, title=f"{prompt_word} Summary")

    agg_fig = plt.figure(figsize=(12, 4))
    agg_axes = agg_fig.subplots(1, 3)
    agg_class_im = agg_axes[0].imshow(
        expanded_aggregation["aggregated_class_logit"], cmap="inferno", vmin=0.0, vmax=1.0
    )
    agg_axes[0].set_title("Expanded Aggregated Class Logit", fontsize=12, fontweight="bold")
    agg_axes[0].axis("off")
    agg_fig.colorbar(agg_class_im, ax=agg_axes[0], fraction=0.046, pad=0.04)

    winner_im = agg_axes[1].imshow(
        expanded_aggregation["winner_map"], cmap="tab10", vmin=0, vmax=max(len(expanded_prompt_artifacts) - 1, 0)
    )
    agg_axes[1].set_title("Expanded Prompt Winner Map", fontsize=12, fontweight="bold")
    agg_axes[1].axis("off")
    agg_fig.colorbar(winner_im, ax=agg_axes[1], fraction=0.046, pad=0.04)

    margin_im = agg_axes[2].imshow(expanded_aggregation["margin_map"], cmap="magma")
    agg_axes[2].set_title("Expanded Prompt Margin\n(top1 - top2)", fontsize=12, fontweight="bold")
    agg_axes[2].axis("off")
    agg_fig.colorbar(margin_im, ax=agg_axes[2], fraction=0.046, pad=0.04)
    agg_fig.suptitle(
        f"{dataset_name} | {image_path.name} | Expanded Prompt Aggregation Details",
        fontsize=16,
        fontweight="bold",
    )
    agg_fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_dir.mkdir(parents=True, exist_ok=True)
    workflow_path = output_dir / f"{image_path.stem}_{class_name.replace(' ', '_')}_workflow.png"
    agg_path = output_dir / f"{image_path.stem}_{class_name.replace(' ', '_')}_aggregation.png"
    fig.suptitle(
        f"{dataset_name} | {image_path.name} | Base vs Expanded Prompt Workflow",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(workflow_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    agg_fig.savefig(agg_path, dpi=200, bbox_inches="tight")
    plt.close(agg_fig)

    return {
        "image": image_path.name,
        "target_class": class_name,
        "workflow_path": str(workflow_path),
        "aggregation_path": str(agg_path),
        "expanded_prompts": expanded_prompt_words,
    }


def main():
    args = parse_args()
    test_data_root = Path(args.test_data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rng = random.Random(args.seed)

    if not test_data_root.exists():
        raise FileNotFoundError(f"Test data root not found: {test_data_root}")

    available_dataset_dirs = sorted(path for path in test_data_root.iterdir() if path.is_dir())
    if args.datasets:
        requested = {name.lower() for name in args.datasets}
        dataset_dirs = [path for path in available_dataset_dirs if path.name.lower() in requested]
    else:
        dataset_dirs = available_dataset_dirs

    if not dataset_dirs:
        raise RuntimeError("No dataset directories selected for visualization.")

    summary: dict[str, list[dict[str, object]]] = {}
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        config_path = SUPPORTED_DATASET_CONFIGS.get(dataset_name)
        if config_path is None:
            print(f"Skipping unsupported dataset folder: {dataset_name}")
            continue

        print(f"Loading base-only model for {dataset_name}")
        base_model, cfg = build_model_from_config(
            config_path,
            device=args.device,
            enable_expanded_prompt=False,
            expanded_prompt_pool_path=None,
        )
        print(f"Loading expanded model for {dataset_name}")
        expanded_model, _ = build_model_from_config(
            config_path,
            device=args.device,
            enable_expanded_prompt=True,
            expanded_prompt_pool_path=resolve_expanded_prompt_pool_path(config_path, cfg),
        )

        class_names = load_base_class_names(cfg.model["classname_path"])
        sampled_images = sample_dataset_images(dataset_dir, args.samples_per_dataset, rng)
        print(
            f"Selected {len(sampled_images)} images for {dataset_name}: "
            + ", ".join(path.name for path in sampled_images)
        )

        dataset_output_dir = output_dir / dataset_name
        summary[dataset_name] = []
        for image_path in sampled_images:
            result = visualize_single_image(
                dataset_name=dataset_name,
                image_path=image_path,
                base_model=base_model,
                expanded_model=expanded_model,
                class_names=class_names,
                output_dir=dataset_output_dir,
                max_expanded_prompts=args.max_expanded_prompts,
            )
            summary[dataset_name].append(result)

        del base_model
        del expanded_model
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Saved visualization summary to {summary_path}")


if __name__ == "__main__":
    main()
