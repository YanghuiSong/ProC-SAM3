from __future__ import annotations

import argparse
import json
import random
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "semantic_instance_fusion_visualizations"
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
            "Visualize how SAM3 semantic HW maps and instance NHW maps are fused "
            "through instance max and semantic-instance max."
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
        help="Number of random images to visualize for each dataset.",
    )
    parser.add_argument(
        "--instance-panels",
        type=int,
        default=4,
        help="Maximum number of top instance maps to render for each selected prompt.",
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


def build_model_from_config(config_path: Path, device: str):
    cfg = Config.fromfile(str(config_path))
    model_cfg = dict(cfg.model)
    model_cfg.pop("type", None)
    model_cfg["device"] = device
    model_cfg["expanded_prompt_pool_path"] = resolve_expanded_prompt_pool_path(
        config_path, cfg
    )
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


def pick_visualization_prompt(model, query_logits: torch.Tensor) -> int:
    query_scores = query_logits.flatten(1).amax(dim=1)
    ranked = torch.argsort(query_scores, descending=True).tolist()
    for prompt_idx in ranked:
        cls_idx = int(model.query_idx[prompt_idx].item())
        if cls_idx != int(model.bg_idx):
            return int(prompt_idx)
    return int(ranked[0])


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


def compute_fusion_artifacts(outputs, output_size: tuple[int, int], confidence_threshold: float):
    height, width = output_size
    eps = 1e-6

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
    instance_margin_hw = instance_max_hw - semantic_hw
    fused_delta_hw = fused_hw - semantic_hw

    return {
        "semantic_hw": semantic_hw,
        "instance_probs_nhw": instance_probs_nhw,
        "weighted_instance_nhw": weighted_instance_nhw,
        "pred_scores": pred_scores.detach().cpu(),
        "valid_scores": valid_scores.detach().cpu(),
        "instance_max_hw": instance_max_hw,
        "fused_hw": fused_hw,
        "source_map": source_map,
        "instance_margin_hw": instance_margin_hw.detach().cpu(),
        "fused_delta_hw": fused_delta_hw.detach().cpu(),
    }


def add_text_panel(ax, lines: list[str], title: str | None = None):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        transform=ax.transAxes,
    )


def visualize_single_image(
    dataset_name: str,
    image_path: Path,
    model,
    class_names: list[str],
    output_dir: Path,
    instance_panels: int,
):
    with Image.open(image_path) as image_handle:
        pil_image = image_handle.convert("RGB")
    image_np = np.array(pil_image)

    with torch.no_grad():
        query_logits = model._inference_single_view(pil_image).detach().cpu()
    prompt_idx = pick_visualization_prompt(model, query_logits)
    prompt_word = model.query_words[prompt_idx]
    cls_idx = int(model.query_idx[prompt_idx].item())
    class_name = class_names[cls_idx]

    outputs = run_single_prompt_raw_outputs(model, pil_image, prompt_idx)
    artifacts = compute_fusion_artifacts(
        outputs=outputs,
        output_size=(pil_image.height, pil_image.width),
        confidence_threshold=model.confidence_threshold,
    )

    weighted_instance_nhw = artifacts["weighted_instance_nhw"]
    pred_scores = artifacts["pred_scores"].numpy()
    valid_scores = artifacts["valid_scores"].numpy().astype(bool)
    top_indices = np.argsort(pred_scores)[::-1][:instance_panels]

    ncols = 6
    nrows = 2 + instance_panels
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 3.0 * nrows),
        squeeze=False,
    )

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    semantic_im = axes[0, 1].imshow(artifacts["semantic_hw"], cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Semantic HW", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")
    fig.colorbar(semantic_im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    instance_max_im = axes[0, 2].imshow(artifacts["instance_max_hw"], cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Instance Max HW", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")
    fig.colorbar(instance_max_im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    fused_im = axes[0, 3].imshow(artifacts["fused_hw"], cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0, 3].set_title("Fused HW = max(semantic, instance)", fontsize=12, fontweight="bold")
    axes[0, 3].axis("off")
    fig.colorbar(fused_im, ax=axes[0, 3], fraction=0.046, pad=0.04)

    source_im = axes[0, 4].imshow(artifacts["source_map"], cmap="coolwarm", vmin=0, vmax=1)
    axes[0, 4].set_title("Pixel Source\n0=semantic, 1=instance", fontsize=12, fontweight="bold")
    axes[0, 4].axis("off")
    fig.colorbar(source_im, ax=axes[0, 4], fraction=0.046, pad=0.04)

    summary_lines = [
        f"Dataset: {dataset_name}",
        f"Image:   {image_path.name}",
        f"Class:   {class_name} (idx={cls_idx})",
        f"Prompt:  q{prompt_idx} -> {prompt_word}",
        "",
        f"Instances (N): {weighted_instance_nhw.shape[0]}",
        f"Valid instances: {int(valid_scores.sum())}",
        f"Conf threshold: {model.confidence_threshold:.2f}",
    ]
    add_text_panel(axes[0, 5], summary_lines, title="Selected Prompt Summary")

    margin_im = axes[1, 0].imshow(artifacts["instance_margin_hw"], cmap="magma")
    axes[1, 0].set_title("Instance Max - Semantic", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    fig.colorbar(margin_im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    delta_im = axes[1, 1].imshow(artifacts["fused_delta_hw"], cmap="viridis")
    axes[1, 1].set_title("Fusion Gain Over Semantic", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")
    fig.colorbar(delta_im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    sem_np = artifacts["semantic_hw"].numpy()
    inst_np = artifacts["instance_max_hw"].numpy()
    fused_np = artifacts["fused_hw"].numpy()
    scatter_lines = [
        f"semantic max: {sem_np.max():.3f}",
        f"instance max: {inst_np.max():.3f}",
        f"fused max:    {fused_np.max():.3f}",
        "",
        f"semantic mean: {sem_np.mean():.3f}",
        f"instance mean: {inst_np.mean():.3f}",
        f"fused mean:    {fused_np.mean():.3f}",
    ]
    add_text_panel(axes[1, 2], scatter_lines, title="Map Statistics")

    valid_lines = []
    for idx in top_indices:
        valid_lines.append(
            f"n{idx:02d} score={pred_scores[idx]:.3f} valid={bool(valid_scores[idx])}"
        )
    add_text_panel(axes[1, 3], valid_lines, title="Top Instance Scores")

    axes[1, 4].imshow(image_np)
    axes[1, 4].set_title("Reference Image", fontsize=12, fontweight="bold")
    axes[1, 4].axis("off")

    axes[1, 5].axis("off")

    for row_offset, instance_idx in enumerate(top_indices, start=2):
        raw_im = axes[row_offset, 0].imshow(
            artifacts["instance_probs_nhw"][instance_idx], cmap="inferno", vmin=0.0, vmax=1.0
        )
        axes[row_offset, 0].set_title(
            f"n{instance_idx}: Raw Instance Prob",
            fontsize=11,
            fontweight="bold",
        )
        axes[row_offset, 0].axis("off")
        fig.colorbar(raw_im, ax=axes[row_offset, 0], fraction=0.046, pad=0.04)

        weighted_im = axes[row_offset, 1].imshow(
            weighted_instance_nhw[instance_idx], cmap="inferno", vmin=0.0, vmax=1.0
        )
        axes[row_offset, 1].set_title(
            f"n{instance_idx}: Weighted Instance",
            fontsize=11,
            fontweight="bold",
        )
        axes[row_offset, 1].axis("off")
        fig.colorbar(weighted_im, ax=axes[row_offset, 1], fraction=0.046, pad=0.04)

        inst_mask = weighted_instance_nhw[instance_idx].numpy()
        dominance = (inst_mask > sem_np).astype(np.int64)
        dominance_im = axes[row_offset, 2].imshow(
            dominance, cmap="coolwarm", vmin=0, vmax=1
        )
        axes[row_offset, 2].set_title(
            f"n{instance_idx}: Beats Semantic",
            fontsize=11,
            fontweight="bold",
        )
        axes[row_offset, 2].axis("off")
        fig.colorbar(dominance_im, ax=axes[row_offset, 2], fraction=0.046, pad=0.04)

        diff_im = axes[row_offset, 3].imshow(
            inst_mask - sem_np, cmap="seismic", vmin=-1.0, vmax=1.0
        )
        axes[row_offset, 3].set_title(
            f"n{instance_idx}: Weighted - Semantic",
            fontsize=11,
            fontweight="bold",
        )
        axes[row_offset, 3].axis("off")
        fig.colorbar(diff_im, ax=axes[row_offset, 3], fraction=0.046, pad=0.04)

        info_lines = [
            f"score={pred_scores[instance_idx]:.3f}",
            f"valid={bool(valid_scores[instance_idx])}",
            f"raw max={float(artifacts['instance_probs_nhw'][instance_idx].max()):.3f}",
            f"weighted max={float(weighted_instance_nhw[instance_idx].max()):.3f}",
            f"winner pixels={(dominance == 1).sum()}",
        ]
        add_text_panel(axes[row_offset, 4], info_lines, title=f"n{instance_idx} Summary")
        axes[row_offset, 5].axis("off")

    fig.suptitle(
        f"{dataset_name} | {image_path.name} | Semantic HW vs Instance NHW Fusion",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_semantic_instance_fusion.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "image": image_path.name,
        "class": class_name,
        "prompt": prompt_word,
        "output_path": str(output_path),
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

    summary: dict[str, list[dict[str, str]]] = {}
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        config_path = SUPPORTED_DATASET_CONFIGS.get(dataset_name)
        if config_path is None:
            print(f"Skipping unsupported dataset folder: {dataset_name}")
            continue

        print(f"Loading model for {dataset_name} from {config_path}")
        model, cfg = build_model_from_config(config_path, device=args.device)
        class_names = cfg.get("class_names") or load_base_class_names(
            cfg.model["classname_path"]
        )

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
                model=model,
                class_names=class_names,
                output_dir=dataset_output_dir,
                instance_panels=args.instance_panels,
            )
            summary[dataset_name].append(result)

        del model
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Saved visualization summary to {summary_path}")


if __name__ == "__main__":
    main()
