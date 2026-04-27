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
from matplotlib import colors as mcolors

import custom_datasets  # noqa: F401
import sam3_segmentor_cached
from mmengine.config import Config


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_DATA_ROOT = REPO_ROOT / "QwSAM3TestData"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "max_aggregation_visualizations"
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
            "Visualize how max aggregation changes prompt-level logits into "
            "category-level logits using a few samples from QwSAM3TestData."
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
        help="Directory used to save the generated comparison figures.",
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
        "--classes-per-image",
        type=int,
        default=2,
        help="Maximum number of multi-prompt classes to visualize per image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling images.",
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


def compute_max_aggregation_artifacts(model, query_logits: torch.Tensor):
    class_logits = torch.full(
        (model.num_cls, *query_logits.shape[-2:]),
        float("-inf"),
        device=query_logits.device,
        dtype=query_logits.dtype,
    )
    winner_maps: dict[int, np.ndarray] = {}
    margin_maps: dict[int, np.ndarray] = {}
    class_prompt_logits: dict[int, torch.Tensor] = {}
    class_prompt_words: dict[int, list[str]] = {}

    query_idx = model.query_idx.to(device=query_logits.device, dtype=torch.long)
    for cls_idx in range(model.num_cls):
        cls_mask = query_idx == cls_idx
        if not cls_mask.any():
            continue

        selected_indices = torch.nonzero(cls_mask, as_tuple=False).flatten()
        cls_logits = query_logits.index_select(0, selected_indices)
        class_prompt_logits[cls_idx] = cls_logits.detach().cpu()
        class_prompt_words[cls_idx] = [model.query_words[int(idx)] for idx in selected_indices]

        max_vals, winner_idx = cls_logits.max(dim=0)
        class_logits[cls_idx] = max_vals
        winner_maps[cls_idx] = winner_idx.detach().cpu().numpy().astype(np.int64)

        if cls_logits.shape[0] > 1:
            top2 = cls_logits.topk(min(2, cls_logits.shape[0]), dim=0).values
            second_vals = top2[1]
        else:
            second_vals = torch.zeros_like(max_vals)
        margin_maps[cls_idx] = (max_vals - second_vals).detach().cpu().numpy()

    return (
        class_logits.detach().cpu(),
        winner_maps,
        margin_maps,
        class_prompt_logits,
        class_prompt_words,
    )


def select_classes_for_visualization(
    model,
    class_logits: torch.Tensor,
    class_prompt_words: dict[int, list[str]],
    max_classes: int,
) -> list[int]:
    class_scores = class_logits.flatten(1).amax(dim=1)
    ranked_indices = torch.argsort(class_scores, descending=True).tolist()

    selected = []
    for cls_idx in ranked_indices:
        if cls_idx == int(model.bg_idx):
            continue
        if len(class_prompt_words.get(cls_idx, [])) <= 1:
            continue
        selected.append(int(cls_idx))
        if len(selected) >= max_classes:
            return selected

    for cls_idx in ranked_indices:
        if cls_idx == int(model.bg_idx):
            continue
        if int(cls_idx) not in selected:
            selected.append(int(cls_idx))
        if len(selected) >= max_classes:
            break
    return selected


def build_palette(num_classes: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20", max(num_classes, 1))
    palette = np.array(
        [np.array(cmap(i)[:3]) * 255 for i in range(num_classes)],
        dtype=np.uint8,
    )
    return palette


def overlay_segmentation(image_np: np.ndarray, pred_mask: np.ndarray, palette: np.ndarray, alpha: float = 0.45):
    color_mask = palette[pred_mask]
    blended = (image_np.astype(np.float32) * (1.0 - alpha) + color_mask.astype(np.float32) * alpha)
    return blended.astype(np.uint8)


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
    classes_per_image: int,
):
    with Image.open(image_path) as image_handle:
        pil_image = image_handle.convert("RGB")

    image_np = np.array(pil_image)
    with torch.no_grad():
        query_logits = model._inference_single_view(pil_image)

    (
        class_logits,
        winner_maps,
        margin_maps,
        class_prompt_logits,
        class_prompt_words,
    ) = compute_max_aggregation_artifacts(model, query_logits)

    pred_mask = class_logits.argmax(dim=0).numpy().astype(np.int64)
    confidence_map = class_logits.max(dim=0).values.numpy()
    selected_classes = select_classes_for_visualization(
        model, class_logits, class_prompt_words, max_classes=classes_per_image
    )

    max_prompt_count = max(
        (len(class_prompt_words.get(cls_idx, [])) for cls_idx in selected_classes),
        default=1,
    )
    ncols = max(4 + max_prompt_count, 5)
    nrows = 1 + len(selected_classes)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.3 * ncols, 3.1 * nrows),
        squeeze=False,
    )

    palette = build_palette(len(class_names))
    overlay_np = overlay_segmentation(image_np, pred_mask, palette)

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(overlay_np)
    axes[0, 1].set_title("Final Segmentation Overlay", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    heatmap_im = axes[0, 2].imshow(confidence_map, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Class Confidence After Max", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")
    fig.colorbar(heatmap_im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    summary_lines = [
        f"Dataset: {dataset_name}",
        f"Image:   {image_path.name}",
        f"Queries: {model.num_queries}",
        f"Classes: {model.num_cls}",
        "",
        "Selected Classes:",
    ]
    for cls_idx in selected_classes:
        class_score = float(class_logits[cls_idx].max().item())
        summary_lines.append(
            f"- {class_names[cls_idx]} (score={class_score:.3f}, prompts={len(class_prompt_words[cls_idx])})"
        )
    add_text_panel(axes[0, 3], summary_lines, title="Comparison Summary")

    for col in range(4, ncols):
        axes[0, col].axis("off")

    for row_idx, cls_idx in enumerate(selected_classes, start=1):
        class_name = class_names[cls_idx]
        prompt_words = class_prompt_words[cls_idx]
        prompt_logits = class_prompt_logits[cls_idx].numpy()
        winner_map = winner_maps[cls_idx]
        margin_map = margin_maps[cls_idx]
        class_map = class_logits[cls_idx].numpy()

        axes[row_idx, 0].imshow(class_map, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[row_idx, 0].set_title(
            f"{class_name}\nAggregated Class Logit",
            fontsize=12,
            fontweight="bold",
        )
        axes[row_idx, 0].axis("off")

        prompt_cmap = plt.get_cmap("tab10", max(len(prompt_words), 1))
        axes[row_idx, 1].imshow(winner_map, cmap=prompt_cmap, vmin=0, vmax=max(len(prompt_words) - 1, 0))
        axes[row_idx, 1].set_title(
            f"{class_name}\nWinning Prompt Index",
            fontsize=12,
            fontweight="bold",
        )
        axes[row_idx, 1].axis("off")

        margin_im = axes[row_idx, 2].imshow(margin_map, cmap="magma", vmin=0.0, vmax=1.0)
        axes[row_idx, 2].set_title(
            f"{class_name}\nMargin (max - second)",
            fontsize=12,
            fontweight="bold",
        )
        axes[row_idx, 2].axis("off")
        fig.colorbar(margin_im, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        winner_lines = []
        total_pixels = max(int(winner_map.size), 1)
        for prompt_idx, prompt_word in enumerate(prompt_words):
            winner_ratio = float((winner_map == prompt_idx).sum()) / total_pixels
            short_prompt = prompt_word if len(prompt_word) <= 30 else prompt_word[:27] + "..."
            winner_lines.append(f"[q{prompt_idx}] {winner_ratio:5.1%}  {short_prompt}")
        add_text_panel(
            axes[row_idx, 3],
            winner_lines,
            title=f"{class_name} Prompt Winners",
        )

        for prompt_col, prompt_word in enumerate(prompt_words, start=4):
            prompt_map = prompt_logits[prompt_col - 4]
            axes[row_idx, prompt_col].imshow(prompt_map, cmap="inferno", vmin=0.0, vmax=1.0)
            short_title = prompt_word if len(prompt_word) <= 26 else prompt_word[:23] + "..."
            axes[row_idx, prompt_col].set_title(
                f"q{prompt_col - 4}: {short_title}",
                fontsize=10,
                fontweight="bold",
            )
            axes[row_idx, prompt_col].axis("off")

        for col in range(4 + len(prompt_words), ncols):
            axes[row_idx, col].axis("off")

    fig.suptitle(
        f"{dataset_name} | {image_path.name} | Max Aggregation Effect",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_max_aggregation.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "image": image_path.name,
        "selected_classes": [class_names[idx] for idx in selected_classes],
        "output_path": str(output_path),
    }


def main():
    args = parse_args()
    test_data_root = Path(args.test_data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rng = random.Random(args.seed)

    if not test_data_root.exists():
        raise FileNotFoundError(f"Test data root not found: {test_data_root}")

    available_dataset_dirs = sorted(
        path for path in test_data_root.iterdir() if path.is_dir()
    )
    if args.datasets:
        requested = {name.lower() for name in args.datasets}
        dataset_dirs = [
            path for path in available_dataset_dirs if path.name.lower() in requested
        ]
    else:
        dataset_dirs = available_dataset_dirs

    if not dataset_dirs:
        raise RuntimeError("No dataset directories selected for visualization.")

    summary: dict[str, list[dict[str, str | list[str]]]] = {}
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

        sampled_images = sample_dataset_images(
            dataset_dir, args.samples_per_dataset, rng
        )
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
                classes_per_image=args.classes_per_image,
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
