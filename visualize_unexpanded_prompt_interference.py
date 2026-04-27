from __future__ import annotations

import argparse
import json
import random
import tempfile
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "unexpanded_prompt_interference_visualizations"
SUPPORTED_DATASET_SPECS = {
    "iSAID": {
        "config": REPO_ROOT / "configs" / "cfg_iSAID_controlled.py",
        "base_class_file": REPO_ROOT / "configs" / "cls_iSAID.txt",
        "exp_class_file": REPO_ROOT / "configs" / "cls_iSAID_exp.txt",
    },
    "LoveDA": {
        "config": REPO_ROOT / "configs" / "cfg_loveda_controlled.py",
        "base_class_file": REPO_ROOT / "configs" / "cls_loveda.txt",
        "exp_class_file": REPO_ROOT / "configs" / "cls_loveda_exp.txt",
    },
    "Potsdam": {
        "config": REPO_ROOT / "configs" / "cfg_potsdam_controlled.py",
        "base_class_file": REPO_ROOT / "configs" / "cls_potsdam.txt",
        "exp_class_file": REPO_ROOT / "configs" / "cls_potsdam_exp.txt",
    },
    "Vaihingen": {
        "config": REPO_ROOT / "configs" / "cfg_vaihingen_controlled.py",
        "base_class_file": REPO_ROOT / "configs" / "cls_vaihingen.txt",
        "exp_class_file": REPO_ROOT / "configs" / "cls_vaihingen_exp.txt",
    },
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize how expanding other classes affects classes whose own prompts "
            "stay unexpanded."
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
        help="Optional subset of dataset names under QwSAM3TestData.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=3,
        help="Number of random images to visualize per dataset.",
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


def normalize_prompt_token(token: str) -> str:
    token = str(token).strip()
    token = token.replace("，", ",").replace("_", " ").replace("-", " ")
    token = " ".join(token.split())
    return token.lower().strip(" ,")


def split_prompt_tokens(text: str) -> list[str]:
    text = str(text).strip()
    if not text:
        return []
    text = text.replace("，", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return parts if parts else [text]


def match_base_class_name(candidate: str, base_classes: list[str]) -> str:
    normalized_candidate = normalize_prompt_token(candidate)
    if not normalized_candidate:
        return candidate

    exact_lookup = {
        normalize_prompt_token(base_class): base_class for base_class in base_classes
    }
    if normalized_candidate in exact_lookup:
        return exact_lookup[normalized_candidate]

    candidate_tokens = set(normalized_candidate.split())
    best_match = candidate
    best_score = 0.0
    best_overlap = 0.0
    best_seq = 0.0
    from difflib import SequenceMatcher

    for base_class in base_classes:
        normalized_base = normalize_prompt_token(base_class)
        base_tokens = set(normalized_base.split())
        token_overlap = len(candidate_tokens & base_tokens) / max(
            len(candidate_tokens | base_tokens), 1
        )
        seq_ratio = SequenceMatcher(None, normalized_candidate, normalized_base).ratio()
        score = 0.65 * seq_ratio + 0.35 * token_overlap
        if score > best_score:
            best_match = base_class
            best_score = score
            best_overlap = token_overlap
            best_seq = seq_ratio

    if best_score >= 0.55:
        return best_match
    if best_overlap > 0.0 and best_seq >= 0.3:
        return best_match
    return candidate


def load_base_alias_map(base_class_file: Path) -> dict[str, list[str]]:
    alias_map: dict[str, list[str]] = {}
    with base_class_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = split_prompt_tokens(line)
            if not parts:
                continue
            canonical = parts[0]
            alias_map[canonical] = parts
    return alias_map


def load_base_class_names(base_class_file: str | Path) -> list[str]:
    alias_map = load_base_alias_map(Path(base_class_file))
    return list(alias_map.keys())


def load_exp_prompt_pool(exp_class_file: Path, base_alias_map: dict[str, list[str]]) -> dict[str, list[str]]:
    base_classes = list(base_alias_map.keys())
    prompt_pool: dict[str, list[str]] = {}
    with exp_class_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = split_prompt_tokens(line)
            if not parts:
                continue
            canonical = match_base_class_name(parts[0], base_classes)
            variants = prompt_pool.setdefault(canonical, [canonical])
            for part in parts[1:]:
                mapped = match_base_class_name(part, base_classes)
                if mapped not in variants:
                    variants.append(mapped)
                if part != mapped and part not in variants:
                    variants.append(part)

    for canonical in base_classes:
        prompt_pool.setdefault(canonical, [canonical])
    return prompt_pool


def find_unexpanded_classes(
    base_alias_map: dict[str, list[str]],
    exp_prompt_pool: dict[str, list[str]],
) -> list[str]:
    unexpanded = []
    for canonical, base_aliases in base_alias_map.items():
        base_norm = {normalize_prompt_token(alias) for alias in base_aliases}
        exp_norm = {
            normalize_prompt_token(variant)
            for variant in exp_prompt_pool.get(canonical, [canonical])
        }
        if exp_norm.issubset(base_norm):
            unexpanded.append(canonical)
    return unexpanded


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


def aggregate_query_logits_by_class(model, query_logits: torch.Tensor) -> torch.Tensor:
    if query_logits.shape[0] == model.num_cls:
        return query_logits.detach().cpu()
    query_idx = model.query_idx.to(device=query_logits.device, dtype=torch.long)
    cls_index = torch.nn.functional.one_hot(
        query_idx, num_classes=model.num_cls
    ).T.view(model.num_cls, len(model.query_idx), 1, 1)
    class_logits = (query_logits.unsqueeze(0) * cls_index).max(1)[0]
    return class_logits.detach().cpu()


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


def select_target_class(
    class_names: list[str],
    unexpanded_classes: list[str],
    base_class_logits: torch.Tensor,
    base_pred: np.ndarray,
    mixed_pred: np.ndarray,
) -> str:
    candidate_scores = []
    for class_name in unexpanded_classes:
        cls_idx = class_names.index(class_name)
        base_pixels = int((base_pred == cls_idx).sum())
        mixed_pixels = int((mixed_pred == cls_idx).sum())
        score = float(base_class_logits[cls_idx].max().item())
        candidate_scores.append((max(base_pixels, mixed_pixels), score, class_name))
    candidate_scores.sort(reverse=True)
    return candidate_scores[0][2]


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


def save_prompt_pool_to_temp(prompt_pool: dict[str, list[str]]) -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    handle.close()
    with open(handle.name, "wb") as file_handle:
        json_safe_pool = {key: list(value) for key, value in prompt_pool.items()}
        import pickle

        pickle.dump(json_safe_pool, file_handle)
    return handle.name


def visualize_single_image(
    dataset_name: str,
    image_path: Path,
    base_model,
    mixed_model,
    class_names: list[str],
    unexpanded_classes: list[str],
    output_dir: Path,
):
    with Image.open(image_path) as image_handle:
        pil_image = image_handle.convert("RGB")
    image_np = np.array(pil_image)

    with torch.no_grad():
        base_query_logits = base_model._inference_single_view(pil_image).detach().cpu()
        mixed_query_logits = mixed_model._inference_single_view(pil_image).detach().cpu()

    base_class_logits = aggregate_query_logits_by_class(base_model, base_query_logits)
    mixed_class_logits = aggregate_query_logits_by_class(mixed_model, mixed_query_logits)

    base_pred = base_class_logits.argmax(dim=0).numpy().astype(np.int64)
    mixed_pred = mixed_class_logits.argmax(dim=0).numpy().astype(np.int64)

    target_class = select_target_class(
        class_names, unexpanded_classes, base_class_logits, base_pred, mixed_pred
    )
    target_cls_idx = class_names.index(target_class)

    base_prompt_idx = int(torch.nonzero(base_model.query_idx == target_cls_idx, as_tuple=False).flatten()[0].item())
    mixed_prompt_idx = int(torch.nonzero(mixed_model.query_idx == target_cls_idx, as_tuple=False).flatten()[0].item())

    base_prompt_logit = base_query_logits[base_prompt_idx].numpy()
    mixed_prompt_logit = mixed_query_logits[mixed_prompt_idx].numpy()
    prompt_diff = mixed_prompt_logit - base_prompt_logit

    base_target_logit = base_class_logits[target_cls_idx].numpy()
    mixed_target_logit = mixed_class_logits[target_cls_idx].numpy()
    class_diff = mixed_target_logit - base_target_logit

    base_target_mask = (base_pred == target_cls_idx).astype(np.int64)
    mixed_target_mask = (mixed_pred == target_cls_idx).astype(np.int64)

    loss_mask = (base_target_mask == 1) & (mixed_target_mask == 0)
    gain_mask = (base_target_mask == 0) & (mixed_target_mask == 1)

    palette = build_palette(len(class_names))
    base_overlay = overlay_segmentation(image_np, base_pred, palette)
    mixed_overlay = overlay_segmentation(image_np, mixed_pred, palette)

    competitor_map = np.full_like(base_pred, fill_value=-1, dtype=np.int64)
    competitor_map[loss_mask] = mixed_pred[loss_mask]
    competitor_counter = Counter(mixed_pred[loss_mask].tolist())
    competitor_lines = []
    for cls_idx, count in competitor_counter.most_common(5):
        competitor_lines.append(f"- {class_names[int(cls_idx)]}: {count} px")
    if not competitor_lines:
        competitor_lines.append("- none")

    change_vis = np.zeros((*base_pred.shape, 3), dtype=np.float32)
    change_vis[..., :] = np.array([0.12, 0.12, 0.12], dtype=np.float32)
    change_vis[loss_mask] = np.array([0.92, 0.25, 0.25], dtype=np.float32)
    change_vis[gain_mask] = np.array([0.25, 0.85, 0.25], dtype=np.float32)

    nrows, ncols = 3, 4
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.6 * nrows),
        squeeze=False,
    )

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(base_overlay)
    axes[0, 1].set_title("Base-Only Segmentation", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mixed_overlay)
    axes[0, 2].set_title("Others Expanded Segmentation", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(change_vis)
    axes[0, 3].set_title(
        "Target Change Map\nred=lost, green=gained",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 3].axis("off")

    prompt_base_im = axes[1, 0].imshow(base_prompt_logit, cmap="inferno")
    axes[1, 0].set_title(
        f"Base Prompt Logit\n{base_model.query_words[base_prompt_idx]}",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].axis("off")
    fig.colorbar(prompt_base_im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    prompt_mixed_im = axes[1, 1].imshow(mixed_prompt_logit, cmap="inferno")
    axes[1, 1].set_title(
        f"Mixed Prompt Logit\n{mixed_model.query_words[mixed_prompt_idx]}",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].axis("off")
    fig.colorbar(prompt_mixed_im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    prompt_diff_im = axes[1, 2].imshow(prompt_diff, cmap="seismic")
    axes[1, 2].set_title("Prompt Logit Diff\n(mixed - base)", fontsize=12, fontweight="bold")
    axes[1, 2].axis("off")
    fig.colorbar(prompt_diff_im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    summary_lines = [
        f"Dataset: {dataset_name}",
        f"Image:   {image_path.name}",
        f"Target class: {target_class}",
        f"Unexpanded classes: {', '.join(unexpanded_classes)}",
        "",
        f"Base prompt count:  {len(torch.nonzero(base_model.query_idx == target_cls_idx, as_tuple=False))}",
        f"Mixed prompt count: {len(torch.nonzero(mixed_model.query_idx == target_cls_idx, as_tuple=False))}",
        f"Base target pixels:  {int(base_target_mask.sum())}",
        f"Mixed target pixels: {int(mixed_target_mask.sum())}",
        f"Lost pixels:         {int(loss_mask.sum())}",
        f"Gained pixels:       {int(gain_mask.sum())}",
        "",
        "Top competitor classes on lost pixels:",
        *competitor_lines,
    ]
    add_text_panel(axes[1, 3], summary_lines, title="Interference Summary")

    class_base_im = axes[2, 0].imshow(base_target_logit, cmap="inferno")
    axes[2, 0].set_title("Base Target Class Logit", fontsize=12, fontweight="bold")
    axes[2, 0].axis("off")
    fig.colorbar(class_base_im, ax=axes[2, 0], fraction=0.046, pad=0.04)

    class_mixed_im = axes[2, 1].imshow(mixed_target_logit, cmap="inferno")
    axes[2, 1].set_title("Mixed Target Class Logit", fontsize=12, fontweight="bold")
    axes[2, 1].axis("off")
    fig.colorbar(class_mixed_im, ax=axes[2, 1], fraction=0.046, pad=0.04)

    class_diff_im = axes[2, 2].imshow(class_diff, cmap="seismic")
    axes[2, 2].set_title("Target Class Logit Diff", fontsize=12, fontweight="bold")
    axes[2, 2].axis("off")
    fig.colorbar(class_diff_im, ax=axes[2, 2], fraction=0.046, pad=0.04)

    competitor_vis = np.zeros((*competitor_map.shape, 3), dtype=np.float32)
    competitor_vis[..., :] = np.array([0.08, 0.08, 0.08], dtype=np.float32)
    for cls_idx in np.unique(competitor_map[competitor_map >= 0]):
        competitor_vis[competitor_map == cls_idx] = palette[int(cls_idx)] / 255.0
    axes[2, 3].imshow(competitor_vis)
    axes[2, 3].set_title(
        "Winning Expanded Competitors\non Target-Lost Pixels",
        fontsize=12,
        fontweight="bold",
    )
    axes[2, 3].axis("off")

    fig.suptitle(
        f"{dataset_name} | {image_path.name} | Unexpanded Prompt Interference",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_{target_class.replace(' ', '_')}_interference.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "image": image_path.name,
        "target_class": target_class,
        "output_path": str(output_path),
        "lost_pixels": int(loss_mask.sum()),
        "gained_pixels": int(gain_mask.sum()),
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

    summary: dict[str, dict[str, object]] = {}
    temp_prompt_paths: list[str] = []
    try:
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            spec = SUPPORTED_DATASET_SPECS.get(dataset_name)
            if spec is None:
                print(f"Skipping unsupported dataset folder: {dataset_name}")
                continue

            base_alias_map = load_base_alias_map(spec["base_class_file"])
            exp_prompt_pool = load_exp_prompt_pool(spec["exp_class_file"], base_alias_map)
            unexpanded_classes = find_unexpanded_classes(base_alias_map, exp_prompt_pool)

            summary[dataset_name] = {
                "unexpanded_classes": unexpanded_classes,
                "results": [],
            }
            if not unexpanded_classes:
                print(f"Skipping {dataset_name}: no unexpanded classes detected in exp file.")
                continue

            mixed_prompt_path = save_prompt_pool_to_temp(exp_prompt_pool)
            temp_prompt_paths.append(mixed_prompt_path)

            print(f"Loading base-only model for {dataset_name}")
            base_model, cfg = build_model_from_config(
                spec["config"],
                device=args.device,
                enable_expanded_prompt=False,
                expanded_prompt_pool_path=None,
            )
            print(f"Loading mixed prompt model for {dataset_name}")
            mixed_model, _ = build_model_from_config(
                spec["config"],
                device=args.device,
                enable_expanded_prompt=True,
                expanded_prompt_pool_path=mixed_prompt_path,
            )

            class_names = load_base_class_names(str(spec["base_class_file"]))

            sampled_images = sample_dataset_images(dataset_dir, args.samples_per_dataset, rng)
            print(
                f"Selected {len(sampled_images)} images for {dataset_name}: "
                + ", ".join(path.name for path in sampled_images)
            )

            dataset_output_dir = output_dir / dataset_name
            for image_path in sampled_images:
                result = visualize_single_image(
                    dataset_name=dataset_name,
                    image_path=image_path,
                    base_model=base_model,
                    mixed_model=mixed_model,
                    class_names=class_names,
                    unexpanded_classes=unexpanded_classes,
                    output_dir=dataset_output_dir,
                )
                summary[dataset_name]["results"].append(result)

            del base_model
            del mixed_model
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"Saved visualization summary to {summary_path}")
    finally:
        for temp_path in temp_prompt_paths:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
