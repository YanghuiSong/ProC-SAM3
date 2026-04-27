from __future__ import annotations

import hashlib
import html
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.qwen_agent import QwenAgent
from palettes import _DATASET_METAINFO
from sam3_segmentor_cached import CachedSAM3OpenSegmentor


try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_NEAREST = Image.NEAREST


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    base_class_file: Path
    expanded_class_file: Path
    palette_key: str
    summary: str
    sample_dir: Path


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    summary: str
    decision_goal: str
    focus_hint: str


@dataclass(frozen=True)
class PromptRequest:
    label: str
    prompt_source: str
    prompt_groups: tuple[tuple[str, ...], ...]
    class_file: Path
    instruction: str = ""

    @property
    def class_count(self) -> int:
        return len(self.prompt_groups)

    @property
    def variant_count(self) -> int:
        return sum(len(group) for group in self.prompt_groups)

    @property
    def class_names(self) -> tuple[str, ...]:
        values: list[str] = []
        for index, group in enumerate(self.prompt_groups):
            values.append(group[0] if group else f"class_{index}")
        return tuple(values)

    @property
    def preview(self) -> str:
        groups = [" | ".join(group) for group in self.prompt_groups[:6]]
        text = " / ".join(groups)
        if len(self.prompt_groups) > 6:
            text += " / ..."
        return text


@dataclass
class ModelSession:
    model: CachedSAM3OpenSegmentor
    lock: threading.Lock


@dataclass
class QwenSession:
    agent: QwenAgent
    lock: threading.Lock


@dataclass
class SegmentationResult:
    dataset_name: str
    prompt_request: PromptRequest
    device: str
    latency_s: float
    cache_hit: bool
    label_map: np.ndarray
    overlay_image: Image.Image
    mask_image: Image.Image
    distribution: list[dict[str, object]]
    dominant_class: dict[str, object] | None
    semantic_ratios: dict[str, float]
    component_counts: dict[str, int]
    pred_instances: object | None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "LoveDA": DatasetSpec(
        name="LoveDA",
        base_class_file=PROJECT_ROOT / "configs" / "cls_loveda.txt",
        expanded_class_file=PROJECT_ROOT / "configs" / "cls_loveda_exp.txt",
        palette_key="LoveDADataset",
        summary="城乡混合区域数据，适合分析建筑、道路、水体、林地、农田与裸地的空间结构。",
        sample_dir=PROJECT_ROOT / "QwSAM3TestData" / "LoveDA",
    ),
    "Vaihingen": DatasetSpec(
        name="Vaihingen",
        base_class_file=PROJECT_ROOT / "configs" / "cls_vaihingen.txt",
        expanded_class_file=PROJECT_ROOT / "configs" / "cls_vaihingen_exp.txt",
        palette_key="ISPRSDataset",
        summary="典型航拍场景，适合分析建筑、道路、低矮植被和树木之间的复杂关系。",
        sample_dir=PROJECT_ROOT / "QwSAM3TestData" / "Vaihingen",
    ),
    "Potsdam": DatasetSpec(
        name="Potsdam",
        base_class_file=PROJECT_ROOT / "configs" / "cls_potsdam.txt",
        expanded_class_file=PROJECT_ROOT / "configs" / "cls_potsdam_exp.txt",
        palette_key="PotsdamDataset",
        summary="高分辨率城市遥感数据，适合观察地物边界、道路骨架与建成区纹理。",
        sample_dir=PROJECT_ROOT / "QwSAM3TestData" / "Potsdam",
    ),
    "iSAID": DatasetSpec(
        name="iSAID",
        base_class_file=PROJECT_ROOT / "configs" / "cls_iSAID.txt",
        expanded_class_file=PROJECT_ROOT / "configs" / "cls_iSAID_exp.txt",
        palette_key="iSAIDDataset",
        summary="密集小目标遥感数据，适合分析飞机、船舶、车辆与场地类目标。",
        sample_dir=PROJECT_ROOT / "QwSAM3TestData" / "iSAID",
    ),
}

SCENARIO_SPECS: dict[str, ScenarioSpec] = {
    "城市与交通治理": ScenarioSpec(
        name="城市与交通治理",
        summary="面向城市更新、道路骨架识别、建成区强度评估与交通组织分析。",
        decision_goal="关注道路、建筑与细粒目标的空间关系，为城市治理和交通研判提供依据。",
        focus_hint="优先观察道路覆盖、建成区比例、车辆密度和蓝绿空间穿插情况。",
    ),
    "环境与人类发展": ScenarioSpec(
        name="环境与人类发展",
        summary="面向生态格局、人居环境、蓝绿空间和硬化地表之间的平衡评估。",
        decision_goal="关注水体、植被、农田与建成区之间的耦合关系，辅助环境治理和人居评价。",
        focus_hint="优先观察蓝绿空间率、水体边界、硬化强度和裸地扰动水平。",
    ),
    "农业与资源监测": ScenarioSpec(
        name="农业与资源监测",
        summary="面向农田分布、资源利用、水系支撑和开发扰动识别。",
        decision_goal="关注农田、水体、裸地和建成干扰，为资源调查和农业监测提供判断依据。",
        focus_hint="优先观察农业覆盖、水体补给、裸地比例和人类建设干扰程度。",
    ),
    "综合巡检与应急保障": ScenarioSpec(
        name="综合巡检与应急保障",
        summary="面向高密建成区巡检、小目标活动识别和复杂地表快速研判。",
        decision_goal="关注建成区、裸地扰动和高频小目标，为巡查、预警和应急部署提供辅助。",
        focus_hint="优先观察细粒目标数量、建成区聚集、裸地扰动和边界复杂区域。",
    ),
}

DEFAULT_DATASET = "LoveDA"
DEFAULT_SCENARIO = "城市与交通治理"
MODEL_CACHE_LIMIT = 4
QWEN_CACHE_LIMIT = 2

SEMANTIC_TAG_ALIASES: dict[str, set[str]] = {
    "building": {"building", "house", "roof", "facade", "structure"},
    "road": {"road", "pavement", "street"},
    "water": {"water", "river", "lake", "sea", "harbor"},
    "vegetation": {"forest", "tree", "vegetation", "grass"},
    "agriculture": {"agricultural", "cropland", "farmland", "field"},
    "bareland": {"barren", "bareland", "soil", "ground"},
    "vehicle": {"vehicle", "car", "truck", "bus", "van"},
    "ship": {"ship", "boat", "vessel"},
    "aircraft": {"plane", "airplane", "aircraft"},
}

APP_CACHE_DIR = PROJECT_ROOT / ".gradio" / "app_cache"
PROMPT_CACHE_DIR = APP_CACHE_DIR / "prompt_files"
FONT_CACHE_DIR = APP_CACHE_DIR / "fonts"
for cache_dir in (APP_CACHE_DIR, PROMPT_CACHE_DIR, FONT_CACHE_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)

_MODEL_CACHE: "OrderedDict[tuple[str, str, str], ModelSession]" = OrderedDict()
_MODEL_CACHE_LOCK = threading.Lock()
_QWEN_CACHE: "OrderedDict[tuple[str, str], QwenSession]" = OrderedDict()
_QWEN_CACHE_LOCK = threading.Lock()
_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

SYSTEM_NAME_SHORT = "天枢遥析"
SYSTEM_NAME_TAGLINE = "基于SAM3与Qwen协同的遥感场景智能解译系统"
SYSTEM_NAME_FULL = f"{SYSTEM_NAME_SHORT}——{SYSTEM_NAME_TAGLINE}"
POSTER_BRAND_NAME = "Tianshu Insight: SAM3 x Qwen Remote Sensing Intelligence"

SYSTEM_SIMHEI_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
BUNDLED_CJK_FONT_PATH = FONT_CACHE_DIR / "simhei.ttf"
if os.name == "nt" and SYSTEM_SIMHEI_PATH.exists() and not BUNDLED_CJK_FONT_PATH.exists():
    try:
        shutil.copyfile(SYSTEM_SIMHEI_PATH, BUNDLED_CJK_FONT_PATH)
    except OSError:
        pass


APP_CSS = """
:root {
  --bg-core: #090b10;
  --panel-bg: rgba(17, 21, 30, 0.68);
  --panel-bg-strong: rgba(13, 17, 23, 0.88);
  --panel-border: rgba(69, 143, 255, 0.14);
  --panel-hover: rgba(69, 143, 255, 0.3);
  --text-main: #e6edf3;
  --text-muted: #8b9bb4;
  --text-soft: #9fb0c8;
  --accent-cyan: #00f0ff;
  --accent-emerald: #00ff66;
  --accent-amber: #ffb000;
  --accent-warn: #ff8f66;
  --shadow-strong: 0 18px 50px rgba(0, 0, 0, 0.42);
  --shadow-glow: 0 0 24px rgba(0, 240, 255, 0.14);
  --font-sans: "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
  --font-display: "Bahnschrift", "Segoe UI", "Microsoft YaHei", sans-serif;
  --font-mono: "JetBrains Mono", Consolas, monospace;
}

body {
  background:
    radial-gradient(circle at 15% 10%, rgba(0, 240, 255, 0.05), transparent 40%),
    radial-gradient(circle at 85% 90%, rgba(0, 255, 102, 0.04), transparent 40%),
    linear-gradient(180deg, var(--bg-core) 0%, #0d1117 100%);
  background-attachment: fixed;
  color: var(--text-main);
  font-family: var(--font-sans);
  overflow-x: hidden;
}

body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(69, 143, 255, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(69, 143, 255, 0.035) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: radial-gradient(circle at center, black 35%, transparent 82%);
  opacity: 0.65;
}

.gradio-container {
  max-width: 1500px !important;
  margin: 0 auto !important;
  padding-top: 18px !important;
  padding-left: 24px !important;
  padding-right: 24px !important;
  width: 100% !important;
  box-sizing: border-box !important;
}

.glass-card {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--panel-border);
  border-radius: 20px;
  background: linear-gradient(180deg, rgba(17, 21, 30, 0.88), rgba(13, 17, 23, 0.82));
  box-shadow: var(--shadow-strong);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}

.glass-card:hover {
  border-color: var(--panel-hover);
  box-shadow: var(--shadow-glow), var(--shadow-strong);
  transform: translateY(-2px);
}

.hero-shell {
  position: relative;
  overflow: hidden;
  margin-bottom: 24px;
  padding: 40px;
  border-radius: 24px;
  border: 1px solid rgba(0, 240, 255, 0.18);
  background: linear-gradient(135deg, rgba(13, 17, 23, 0.9), rgba(22, 27, 34, 0.85));
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), inset 0 0 0 1px rgba(255, 255, 255, 0.04);
}

.hero-shell::after {
  content: "";
  position: absolute;
  right: -120px;
  top: -120px;
  width: 320px;
  height: 320px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0, 240, 255, 0.08), transparent 70%);
  filter: blur(10px);
}

.hero-grid {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 40px;
  align-items: center;
}

.hero-copy {
  min-width: 0;
}

.hero-eyebrow {
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--accent-cyan);
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 12px;
  text-shadow: 0 0 8px rgba(0, 240, 255, 0.4);
}

.hero-title {
  margin: 0;
  font-family: var(--font-display);
  font-size: 48px;
  font-weight: 700;
  line-height: 1.08;
  letter-spacing: -1px;
  background: linear-gradient(90deg, #ffffff, #8b9bb4);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero-subtitle {
  margin: 20px 0 24px;
  color: var(--text-muted);
  font-size: 16px;
  line-height: 1.7;
  font-weight: 300;
  max-width: 760px;
}

.hero-badge-row,
.legend-wrap,
.chip-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.hero-badge,
.legend-chip,
.scenario-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(0, 0, 0, 0.38);
  color: var(--text-main);
  font-size: 13px;
  font-weight: 600;
  transition: all 0.3s ease;
}

.hero-badge:hover,
.legend-chip:hover,
.scenario-pill:hover {
  border-color: var(--accent-cyan);
  background: rgba(0, 240, 255, 0.05);
  transform: translateY(-2px);
}

.hero-orbit {
  position: relative;
  display: grid;
  align-items: center;
  justify-content: center;
  min-height: 320px;
}

.orbit-stage {
  position: absolute;
  inset: 0;
  border-radius: 50%;
  border: 1px solid rgba(69, 143, 255, 0.1);
  background: radial-gradient(circle, rgba(0, 240, 255, 0.03) 0%, transparent 70%);
}

.orbit-stage::after {
  content: "";
  position: absolute;
  inset: -10%;
  border-radius: 50%;
  background: conic-gradient(from 0deg, transparent 0deg, rgba(0, 240, 255, 0.22) 60deg, transparent 96deg);
  animation: radar-sweep 4s linear infinite;
  mask-image: radial-gradient(circle, transparent 30%, black 70%);
}

.orbit-ring {
  position: absolute;
  inset: 50%;
  border-radius: 50%;
  border: 1px dashed rgba(0, 240, 255, 0.25);
  transform: translate(-50%, -50%);
}

.orbit-ring-a {
  width: 82%;
  height: 82%;
  animation: spin-slow 30s linear infinite;
}

.orbit-ring-b {
  width: 54%;
  height: 54%;
  border: 1px solid rgba(0, 255, 102, 0.2);
  animation: spin-slow 20s linear infinite reverse;
}

.orbit-core {
  position: absolute;
  left: 50%;
  top: 50%;
  width: 60px;
  height: 60px;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  background: radial-gradient(circle, var(--accent-cyan), transparent);
  box-shadow: 0 0 30px var(--accent-cyan);
  animation: pulse-core 2s ease-in-out infinite;
}

.orbit-label {
  position: absolute;
  padding: 6px 12px;
  border-radius: 6px;
  background: rgba(13, 17, 23, 0.82);
  border: 1px solid rgba(69, 143, 255, 0.3);
  color: var(--accent-cyan);
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 1px;
  text-transform: uppercase;
}

.orbit-label.urban {
  left: 4%;
  top: 16%;
}

.orbit-label.water {
  right: 5%;
  top: 22%;
  color: var(--accent-emerald);
  border-color: rgba(0, 255, 102, 0.3);
}

.orbit-label.vegetation {
  left: 10%;
  bottom: 15%;
  color: var(--accent-amber);
  border-color: rgba(255, 176, 0, 0.3);
}

.metric-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 16px;
  margin: 20px 0 0;
}

.metric-card {
  padding: 16px;
  border-radius: 12px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.05);
  transition: all 0.3s ease;
}

.metric-card:hover {
  border-color: rgba(0, 240, 255, 0.4);
  background: rgba(0, 240, 255, 0.02);
}

.metric-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-muted);
  font-family: var(--font-mono);
}

.metric-value {
  font-size: 28px;
  font-weight: 700;
  color: #fff;
  margin-top: 8px;
  font-family: var(--font-sans);
  line-height: 1.12;
}

.metric-note {
  color: rgba(139, 155, 180, 0.74);
  font-size: 12px;
  margin-top: 6px;
  line-height: 1.5;
}

.section-shell {
  padding: 24px;
}

.section-title {
  font-size: 20px;
  font-weight: 700;
  color: #fff;
  margin: 0 0 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.section-title::before {
  content: "";
  display: block;
  width: 4px;
  height: 18px;
  background: var(--accent-cyan);
  border-radius: 2px;
  box-shadow: 0 0 8px var(--accent-cyan);
}

.section-copy,
.status-copy {
  color: var(--text-muted);
  font-size: 14px;
  line-height: 1.65;
}

.scenario-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-top: 18px;
}

.scene-chip {
  padding: 16px 12px;
  border-radius: 12px;
  color: white;
  min-height: 100px;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  border: 1px solid rgba(255,255,255,0.1);
  position: relative;
  overflow: hidden;
}

.scene-chip::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, transparent, rgba(0,0,0,0.82));
  z-index: 0;
}

.scene-chip > * {
  position: relative;
  z-index: 1;
}

.scene-chip span:first-child {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: rgba(255,255,255,0.72);
  font-family: var(--font-mono);
}

.scene-chip span:last-child {
  margin-top: 4px;
  font-size: 15px;
  font-weight: 700;
}

.scene-chip.urban {
  background: linear-gradient(135deg, #27394d, #1a2530);
}

.scene-chip.water {
  background: linear-gradient(135deg, #0b3d91, #08244f);
}

.scene-chip.veg {
  background: linear-gradient(135deg, #1b4d3e, #0f2f24);
}

.scene-chip.road {
  background: linear-gradient(135deg, #5c4033, #2f211b);
}

.legend-wrap {
  margin-top: 10px;
}

.legend-chip {
  border-radius: 999px;
}

.legend-swatch {
  width: 12px;
  height: 12px;
  border-radius: 4px;
}

.insight-list {
  display: grid;
  gap: 10px;
  margin-top: 14px;
}

.insight-item {
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(111, 215, 255, 0.09);
  background: rgba(255, 255, 255, 0.03);
  color: var(--text-soft);
  line-height: 1.72;
}

.result-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 16px;
  font-size: 14px;
}

.result-table th,
.result-table td {
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}

.result-table th {
  color: var(--text-muted);
  font-weight: 600;
  font-family: var(--font-mono);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.result-table td {
  color: var(--text-soft);
}

.result-table tr:hover td {
  background: rgba(255,255,255,0.02);
}

.status-box {
  padding: 16px 18px;
  border-radius: 16px;
  border: 1px solid var(--panel-border);
  background: rgba(0, 0, 0, 0.34);
}

.status-title {
  font-weight: 700;
  font-size: 16px;
  color: #fff;
  margin-bottom: 8px;
}

.warn {
  border-color: rgba(255, 176, 0, 0.3);
}

.gradio-container button {
  border-radius: 8px !important;
  font-weight: 600 !important;
  letter-spacing: 0.4px;
  transition: all 0.3s ease !important;
}

.gradio-container button.primary {
  background: linear-gradient(135deg, #0055ff, #00f0ff) !important;
  color: #000 !important;
  border: none !important;
  box-shadow: 0 4px 14px rgba(0, 240, 255, 0.3) !important;
}

.gradio-container button.primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 240, 255, 0.5) !important;
}

.gradio-container button.secondary {
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  color: var(--text-main) !important;
}

.gradio-container button.secondary:hover {
  background: rgba(255, 255, 255, 0.1) !important;
  border-color: rgba(255, 255, 255, 0.2) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .gr-box,
.gradio-container .wrap,
.gradio-container .form {
  background: rgba(0, 0, 0, 0.4) !important;
  border-color: rgba(255, 255, 255, 0.1) !important;
  color: white !important;
  border-radius: 8px !important;
  font-family: var(--font-sans) !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus {
  border-color: var(--accent-cyan) !important;
  box-shadow: 0 0 0 1px var(--accent-cyan) !important;
}

.gradio-container [role="tablist"] {
  border-bottom: 1px solid rgba(255,255,255,0.1) !important;
  background: transparent !important;
}

.gradio-container [role="tab"] {
  border: none !important;
  background: transparent !important;
  color: var(--text-muted) !important;
  font-weight: 600 !important;
  padding: 12px 24px !important;
  border-bottom: 2px solid transparent !important;
}

.gradio-container [role="tab"][aria-selected="true"] {
  color: var(--accent-cyan) !important;
  border-bottom-color: var(--accent-cyan) !important;
  background: linear-gradient(0deg, rgba(0,240,255,0.05), transparent) !important;
}

.gradio-container .gallery-item {
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  background: rgba(0,0,0,0.5) !important;
  overflow: hidden;
}

@keyframes radar-sweep {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

@keyframes spin-slow {
  from {
    transform: translate(-50%, -50%) rotate(0deg);
  }
  to {
    transform: translate(-50%, -50%) rotate(360deg);
  }
}

@keyframes pulse-core {
  0%, 100% {
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.5);
  }
  50% {
    box-shadow: 0 0 40px rgba(0, 240, 255, 0.8), 0 0 80px rgba(0, 240, 255, 0.3);
  }
}

@media (max-width: 1100px) {
  .hero-grid {
    grid-template-columns: 1fr;
  }

  .scenario-strip {
    grid-template-columns: repeat(2, 1fr);
  }

  .metric-row {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .hero-shell {
    padding: 24px 20px;
  }

  .hero-title {
    font-size: 34px;
  }

  .scenario-strip,
  .metric-row {
    grid-template-columns: 1fr;
  }
}
"""


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    cache_key = (size, bold)
    cached = _FONT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    candidates: list[Path] = [BUNDLED_CJK_FONT_PATH]
    if os.name == "nt":
        candidates.extend(
            [
                SYSTEM_SIMHEI_PATH,
                Path(r"C:\Windows\Fonts\msyhbd.ttc") if bold else Path(r"C:\Windows\Fonts\msyh.ttc"),
                Path(r"C:\Windows\Fonts\Dengb.ttf") if bold else Path(r"C:\Windows\Fonts\Deng.ttf"),
                Path(r"C:\Windows\Fonts\simsunb.ttf") if bold else Path(r"C:\Windows\Fonts\simsun.ttc"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
                if bold
                else Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            ]
        )

    for path in candidates:
        if path.exists():
            try:
                font = ImageFont.truetype(str(path), size=size)
                _FONT_CACHE[cache_key] = font
                return font
            except OSError:
                continue

    font = ImageFont.load_default()
    _FONT_CACHE[cache_key] = font
    return font


def get_available_datasets() -> list[str]:
    return list(DATASET_SPECS.keys())


def get_available_scenarios() -> list[str]:
    return list(SCENARIO_SPECS.keys())


def get_palette_rgb(dataset_name: str) -> np.ndarray:
    palette = _DATASET_METAINFO[DATASET_SPECS[dataset_name].palette_key]["palette"]
    return np.asarray(palette, dtype=np.uint8)


def get_dataset_classes(dataset_name: str) -> list[str]:
    return list(_DATASET_METAINFO[DATASET_SPECS[dataset_name].palette_key]["classes"])


def list_sample_images(dataset_name: str) -> tuple[str, ...]:
    sample_dir = DATASET_SPECS[dataset_name].sample_dir
    if not sample_dir.exists():
        return ()
    values = sorted(str(path) for path in sample_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"})
    return tuple(values)


def get_available_devices() -> list[str]:
    if not torch.cuda.is_available():
        return ["cpu"]
    return [f"cuda:{index}" for index in range(torch.cuda.device_count())]


def get_default_sam_device() -> str:
    devices = get_available_devices()
    return devices[0]


def get_default_qwen_device() -> str:
    devices = get_available_devices()
    if len(devices) >= 2:
        return devices[1]
    return devices[0]


def normalize_device_choice(device: str) -> str:
    choices = get_available_devices()
    if device in choices:
        return device
    return choices[0]


def render_metrics_row(items: list[tuple[str, str, str]]) -> str:
    blocks = []
    for label, value, note in items:
        blocks.append(
            "<div class='metric-card'>"
            f"<div class='metric-label'>{html.escape(label)}</div>"
            f"<div class='metric-value'>{html.escape(value)}</div>"
            f"<div class='metric-note'>{html.escape(note)}</div>"
            "</div>"
        )
    return f"<div class='metric-row'>{''.join(blocks)}</div>"


def parse_results_file(file_path: Path) -> list[dict[str, float]]:
    if not file_path.exists():
        return []
    records: list[dict[str, float]] = []
    current: dict[str, float] = {}
    for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            if current:
                records.append(current)
                current = {}
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key in {"aAcc", "mIoU", "mAcc", "AP50", "AP75"}:
            try:
                current[key] = float(value)
            except ValueError:
                continue
    if current:
        records.append(current)
    return records


def get_best_metrics(file_path: Path) -> dict[str, float]:
    records = parse_results_file(file_path)
    if not records:
        return {}
    best: dict[str, float] = {}
    for key in ("mIoU", "AP50", "AP75"):
        values = [record.get(key) for record in records if key in record]
        if values:
            best[key] = max(values)
    return best


def build_experiment_snapshot() -> dict[str, dict[str, float]]:
    snapshot: dict[str, dict[str, float]] = {}
    snapshot["Vaihingen_base"] = get_best_metrics(PROJECT_ROOT / "work_dirs" / "cfg_vaihingen" / "results.txt")
    snapshot["Vaihingen_expanded"] = get_best_metrics(PROJECT_ROOT / "work_dirs" / "cfg_vaihingen_exp" / "results.txt")
    snapshot["LoveDA"] = get_best_metrics(PROJECT_ROOT / "work_dirs" / "cfg_loveda_controlled" / "results.txt")
    snapshot["iSAID"] = get_best_metrics(PROJECT_ROOT / "work_dirs" / "cfg_iSAID_exp" / "results.txt")
    return snapshot


EXPERIMENT_SNAPSHOT = build_experiment_snapshot()


def _metric_text(values: dict[str, float], key: str) -> str:
    value = values.get(key)
    return f"{value:.2f}" if value is not None else "N/A"


def build_header_html() -> str:
    vai_base = EXPERIMENT_SNAPSHOT.get("Vaihingen_base", {})
    vai_exp = EXPERIMENT_SNAPSHOT.get("Vaihingen_expanded", {})
    loveda = EXPERIMENT_SNAPSHOT.get("LoveDA", {})
    isaid = EXPERIMENT_SNAPSHOT.get("iSAID", {})
    gain = vai_exp.get("mIoU", 0.0) - vai_base.get("mIoU", 0.0) if vai_base and vai_exp else 0.0

    experiment_cards = render_metrics_row(
        [
            ("Vaihingen 基础", _metric_text(vai_base, "mIoU"), "最佳 mIoU"),
            ("Vaihingen 扩展", _metric_text(vai_exp, "mIoU"), f"最佳 mIoU，提升 {gain:.2f}"),
            ("LoveDA", _metric_text(loveda, "mIoU"), f"AP50 {_metric_text(loveda, 'AP50')}"),
            ("iSAID", _metric_text(isaid, "mIoU"), f"AP50 {_metric_text(isaid, 'AP50')}"),
        ]
    )
    return (
        "<div class='hero-shell'>"
        "<div class='hero-grid'>"
        "<div class='hero-copy'>"
        "<div class='hero-eyebrow'>SAM3 × Qwen 协同解译引擎</div>"
        f"<h1 class='hero-title'>{SYSTEM_NAME_SHORT}</h1>"
        "<div class='hero-subtitle'>"
        f"以 <strong>{SYSTEM_NAME_SHORT}</strong> 为核心品牌，整体系统定位为“{SYSTEM_NAME_TAGLINE}”。"
        "界面聚焦遥感分割、场景研判、结构化统计与答辩级成果展示，"
        "让模型能力、解释链路与业务价值能够在同一页面中自然呈现。"
        "</div>"
        "<div class='hero-badge-row'>"
        "<span class='hero-badge'>SAM3 遥感分割</span>"
        "<span class='hero-badge'>Qwen 场景评估</span>"
        "<span class='hero-badge'>结构化指标提取</span>"
        "<span class='hero-badge'>批量统计与展示</span>"
        "</div>"
        "</div>"
        "<div class='hero-orbit'>"
        "<div class='orbit-stage'></div>"
        "<div class='orbit-ring orbit-ring-a'></div>"
        "<div class='orbit-ring orbit-ring-b'></div>"
        "<div class='orbit-core'></div>"
        "<div class='orbit-label urban'>Urban Grid</div>"
        "<div class='orbit-label water'>Hydro System</div>"
        "<div class='orbit-label vegetation'>Eco Scan</div>"
        "</div>"
        "</div>"
        f"{experiment_cards}"
        "<div class='status-box' style='margin-top: 18px;'>"
        "<div class='status-title'>展示逻辑</div>"
        "<div class='status-copy'>"
        "1. 先做遥感分割，再把面积占比、主导地物和目标密度转成决策指标。"
        "2. Qwen 只负责结合所选场景解释结果，不评价提示词好坏。"
        "3. 海报、统计卡和批量面板统一服务于比赛答辩展示。"
        "</div>"
        "</div>"
        "</div>"
    )


def build_dataset_overview_html(dataset_name: str) -> str:
    spec = DATASET_SPECS[dataset_name]
    classes = get_dataset_classes(dataset_name)
    palette = get_palette_rgb(dataset_name)
    sample_count = len(list_sample_images(dataset_name))

    legend_items = []
    for name, color in zip(classes, palette):
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        swatch_style = f"background: rgb({r}, {g}, {b}); box-shadow: 0 0 6px rgb({r}, {g}, {b});"
        legend_items.append(
            "<span class='legend-chip'>"
            f"<span class='legend-swatch' style='{swatch_style}'></span>{html.escape(name)}"
            "</span>"
        )

    experiment_note = "未找到对应实验记录。"
    if dataset_name == "Vaihingen":
        base_values = EXPERIMENT_SNAPSHOT.get("Vaihingen_base", {})
        exp_values = EXPERIMENT_SNAPSHOT.get("Vaihingen_expanded", {})
        if base_values and exp_values:
            experiment_note = (
                f"已有实验中，标准词表最佳 mIoU 为 {_metric_text(base_values, 'mIoU')}，"
                f"增强词表最佳 mIoU 为 {_metric_text(exp_values, 'mIoU')}。"
            )
    elif dataset_name in EXPERIMENT_SNAPSHOT and EXPERIMENT_SNAPSHOT[dataset_name]:
        values = EXPERIMENT_SNAPSHOT[dataset_name]
        experiment_note = f"已有实验最佳 mIoU 为 {_metric_text(values, 'mIoU')}，AP50 为 {_metric_text(values, 'AP50')}。"

    metrics_html = render_metrics_row(
        [
            ("地物类别数", str(len(classes)), "当前调色板类别"),
            ("示例图像数", str(sample_count), "本地测试样本"),
            ("标准词表变体", str(sum(len(group) for group in load_prompt_groups(str(spec.base_class_file)))), "基础分割词表"),
            ("增强词表变体", str(sum(len(group) for group in load_prompt_groups(str(spec.expanded_class_file)))), "增强分割词表"),
        ]
    )
    return (
        "<div class='glass-card section-shell'>"
        "<div class='section-title'>数据集信息</div>"
        f"<div class='section-copy'><strong>{html.escape(spec.name)}</strong>：{html.escape(spec.summary)}</div>"
        f"<div class='section-copy'><strong>实验依据：</strong>{html.escape(experiment_note)}</div>"
        "<div class='scenario-strip'>"
        "<div class='scene-chip urban'><span>Urban</span><span>建筑与建成区</span></div>"
        "<div class='scene-chip water'><span>Hydrology</span><span>水体与边界</span></div>"
        "<div class='scene-chip veg'><span>Eco Layer</span><span>植被与农田</span></div>"
        "<div class='scene-chip road'><span>Transport</span><span>道路与骨架</span></div>"
        "</div>"
        f"{metrics_html}"
        "<div class='section-copy' style='margin-top: 14px;'>类别图例</div>"
        f"<div class='legend-wrap'>{''.join(legend_items)}</div>"
        "</div>"
    )


def prompt_groups_to_text(prompt_groups: Iterable[Iterable[str]]) -> str:
    return "\n".join(",".join(group) for group in prompt_groups)


def prompt_groups_digest(prompt_groups: Iterable[Iterable[str]]) -> str:
    return hashlib.sha1(prompt_groups_to_text(prompt_groups).encode("utf-8")).hexdigest()[:16]


def load_prompt_groups(file_path: str) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    path = Path(file_path)
    if not path.exists():
        return ()
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = tuple(token.strip() for token in line.split(",") if token.strip())
        if parts:
            groups.append(parts)
    return tuple(groups)


def parse_custom_prompt_groups(custom_prompts: str) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for chunk in re.split(r"[\n;,；，]+", custom_prompts or ""):
        item = chunk.strip()
        if not item:
            continue
        variants = tuple(part.strip() for part in item.split("|") if part.strip())
        if variants:
            groups.append(variants)
    return tuple(groups)


def ensure_background_prompt_group(prompt_groups: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
    if not prompt_groups:
        return prompt_groups
    first_tokens = {group[0].strip().lower() for group in prompt_groups if group}
    if {"background", "bg"} & first_tokens:
        return prompt_groups
    return (("background",),) + prompt_groups


def create_prompt_file(prompt_groups: tuple[tuple[str, ...], ...], dataset_name: str) -> Path:
    normalized_groups = ensure_background_prompt_group(prompt_groups)
    file_name = f"{dataset_name}_{prompt_groups_digest(normalized_groups)}.txt"
    path = PROMPT_CACHE_DIR / file_name
    if not path.exists():
        path.write_text(prompt_groups_to_text(normalized_groups), encoding="utf-8")
    return path


def build_prompt_request(
    dataset_name: str,
    prompt_mode: str,
    custom_prompts: str,
) -> PromptRequest:
    dataset = DATASET_SPECS[dataset_name]
    custom_groups = parse_custom_prompt_groups(custom_prompts)
    if custom_groups:
        normalized_groups = ensure_background_prompt_group(custom_groups)
        return PromptRequest(
            label="自定义词表",
            prompt_source="custom",
            prompt_groups=normalized_groups,
            class_file=create_prompt_file(normalized_groups, dataset_name),
        )

    normalized_mode = {"基础提示词": "标准词表", "扩展提示词": "增强词表"}.get(prompt_mode, prompt_mode)
    if normalized_mode == "增强词表":
        class_file = dataset.expanded_class_file
        label = "增强词表"
        prompt_source = "expanded"
    else:
        class_file = dataset.base_class_file
        label = "标准词表"
        prompt_source = "base"

    groups = ensure_background_prompt_group(load_prompt_groups(str(class_file)))
    return PromptRequest(
        label=label,
        prompt_source=prompt_source,
        prompt_groups=groups,
        class_file=class_file,
    )


def build_prompt_preview_text(
    dataset_name: str,
    prompt_mode: str,
    custom_prompts: str,
    enable_qwen_analysis: bool,
    scenario_name: str,
) -> str:
    request = build_prompt_request(dataset_name, prompt_mode, custom_prompts)
    scenario = SCENARIO_SPECS.get(scenario_name, SCENARIO_SPECS[DEFAULT_SCENARIO])
    lines = [
        f"当前分割方案：{request.label}",
        f"类别数：{request.class_count} | 词表变体数：{request.variant_count}",
        f"当前评估场景：{scenario.name}",
        "-" * 40,
    ]
    lines.extend(f"> {' | '.join(group)}" for group in request.prompt_groups[:10])
    if request.class_count > 10:
        lines.append("> ...")
    if enable_qwen_analysis:
        lines.extend(
            [
                "",
                "Qwen 场景评估会读取：",
                f"- 场景目标：{scenario.decision_goal}",
                f"- 关注重点：{scenario.focus_hint}",
                "- 分割词表只用于推理，不作为评价对象",
            ]
        )
    return "\n".join(lines)


def get_model_session(prompt_request: PromptRequest, device: str) -> tuple[ModelSession, bool]:
    device_key = normalize_device_choice(device)
    cache_key = (prompt_request.prompt_source, str(prompt_request.class_file), device_key)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            _MODEL_CACHE.move_to_end(cache_key)
            return cached, True

    session = ModelSession(
        model=CachedSAM3OpenSegmentor(
            classname_path=str(prompt_request.class_file),
            device=device_key,
            prob_thd=0.1,
            confidence_threshold=0.4,
            use_sem_seg=True,
            use_presence_score=True,
            use_transformer_decoder=True,
        ),
        lock=threading.Lock(),
    )

    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = session
        _MODEL_CACHE.move_to_end(cache_key)
        if len(_MODEL_CACHE) > MODEL_CACHE_LIMIT:
            _MODEL_CACHE.popitem(last=False)
    return session, False


def get_qwen_session(dataset_name: str, device: str) -> tuple[QwenSession, bool]:
    device_key = normalize_device_choice(device)
    cache_key = (dataset_name, device_key)
    with _QWEN_CACHE_LOCK:
        cached = _QWEN_CACHE.get(cache_key)
        if cached is not None:
            _QWEN_CACHE.move_to_end(cache_key)
            return cached, True

    session = QwenSession(
        agent=QwenAgent(device=device_key, dataset_name=dataset_name),
        lock=threading.Lock(),
    )
    with _QWEN_CACHE_LOCK:
        _QWEN_CACHE[cache_key] = session
        _QWEN_CACHE.move_to_end(cache_key)
        if len(_QWEN_CACHE) > QWEN_CACHE_LIMIT:
            _QWEN_CACHE.popitem(last=False)
    return session, False


def clear_model_cache() -> int:
    qwen_count = 0
    with _MODEL_CACHE_LOCK:
        count = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()
    with _QWEN_CACHE_LOCK:
        qwen_count = len(_QWEN_CACHE)
        _QWEN_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return count + qwen_count


def aggregate_query_logits_to_class_logits(
    seg_logits: torch.Tensor,
    query_idx: torch.Tensor | np.ndarray | list[int],
    num_cls: int | torch.Tensor,
) -> torch.Tensor:
    if isinstance(num_cls, torch.Tensor):
        num_cls = int(num_cls.detach().cpu().item())
    if isinstance(query_idx, torch.Tensor):
        query_idx_tensor = query_idx.to(device=seg_logits.device, dtype=torch.long)
    else:
        query_idx_tensor = torch.tensor(query_idx, device=seg_logits.device, dtype=torch.long)

    if seg_logits.shape[0] == num_cls and query_idx_tensor.numel() == num_cls:
        return seg_logits

    cls_index = torch.nn.functional.one_hot(query_idx_tensor, num_classes=int(num_cls))
    cls_index = cls_index.T.view(int(num_cls), query_idx_tensor.numel(), 1, 1).to(seg_logits.device)
    return (seg_logits.unsqueeze(0) * cls_index).max(1)[0]


def compute_label_boundaries(label_map: np.ndarray) -> np.ndarray:
    boundaries = np.zeros(label_map.shape, dtype=bool)
    boundaries[:-1, :] |= label_map[:-1, :] != label_map[1:, :]
    boundaries[1:, :] |= label_map[:-1, :] != label_map[1:, :]
    boundaries[:, :-1] |= label_map[:, :-1] != label_map[:, 1:]
    boundaries[:, 1:] |= label_map[:, :-1] != label_map[:, 1:]
    return boundaries


def colorize_prediction(label_map: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    safe_map = label_map.copy().astype(np.int64)
    invalid_mask = (safe_map < 0) | (safe_map >= len(palette_rgb))
    safe_map[invalid_mask] = 0
    color_mask = palette_rgb[safe_map].astype(np.uint8)
    color_mask[invalid_mask] = np.array([0, 0, 0], dtype=np.uint8)
    return color_mask


def blend_images(original_rgb: np.ndarray, mask_rgb: np.ndarray, label_map: np.ndarray, alpha: float) -> tuple[Image.Image, Image.Image]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = (original_rgb * (1.0 - alpha) + mask_rgb * alpha).astype(np.uint8)
    boundaries = compute_label_boundaries(label_map)
    overlay[boundaries] = np.array([63, 208, 255], dtype=np.uint8)
    mask_rgb = mask_rgb.copy()
    mask_rgb[boundaries] = np.array([63, 208, 255], dtype=np.uint8)
    return Image.fromarray(overlay), Image.fromarray(mask_rgb)


def summarize_distribution(label_map: np.ndarray, class_names: Iterable[str], palette_rgb: np.ndarray) -> list[dict[str, object]]:
    labels, counts = np.unique(label_map, return_counts=True)
    names = list(class_names)
    total_pixels = max(int(label_map.size), 1)
    summary: list[dict[str, object]] = []
    for label, count in zip(labels.tolist(), counts.tolist()):
        name = names[label] if 0 <= label < len(names) else f"class_{label}"
        color = tuple(int(channel) for channel in (palette_rgb[label] if label < len(palette_rgb) else [0, 0, 0]))
        summary.append(
            {
                "index": int(label),
                "name": name,
                "pixels": int(count),
                "ratio": float(count / total_pixels),
                "color": color,
            }
        )
    summary.sort(key=lambda item: item["ratio"], reverse=True)
    return summary


def get_dominant_foreground_class(distribution: list[dict[str, object]]) -> dict[str, object] | None:
    for item in distribution:
        if str(item["name"]).lower() not in {"background", "bg"}:
            return item
    return distribution[0] if distribution else None


def normalize_semantic_token(text: str) -> str:
    token = str(text or "").strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", token)


def match_semantic_tag(label: str) -> str:
    normalized = normalize_semantic_token(label)
    for tag, aliases in SEMANTIC_TAG_ALIASES.items():
        if normalized in aliases:
            return tag
        if any(alias in normalized for alias in aliases):
            return tag
    return normalized or "unknown"


def aggregate_distribution_by_semantic_tag(distribution: list[dict[str, object]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in distribution:
        tag = match_semantic_tag(str(item["name"]))
        values[tag] = values.get(tag, 0.0) + float(item["ratio"])
    return values


def estimate_connected_components(binary_mask: np.ndarray, min_pixels: int = 6) -> int:
    if binary_mask.size == 0:
        return 0
    visited = np.zeros(binary_mask.shape, dtype=bool)
    count = 0
    h, w = binary_mask.shape
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y in range(h):
        for x in range(w):
            if not binary_mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for oy, ox in offsets:
                    ny, nx = cy + oy, cx + ox
                    if 0 <= ny < h and 0 <= nx < w and binary_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area >= min_pixels:
                count += 1
    return count


def estimate_component_counts(label_map: np.ndarray, class_names: Iterable[str]) -> dict[str, int]:
    height, width = label_map.shape
    max_side = max(height, width)
    if max_side > 192:
        scale = 192.0 / float(max_side)
        reduced = Image.fromarray(label_map.astype(np.uint8), mode="L").resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            resample=RESAMPLE_NEAREST,
        )
        reduced_map = np.asarray(reduced, dtype=np.uint8)
    else:
        reduced_map = label_map.astype(np.uint8)

    tag_to_indices: dict[str, list[int]] = {}
    for index, name in enumerate(class_names):
        tag_to_indices.setdefault(match_semantic_tag(name), []).append(index)

    counts: dict[str, int] = {}
    for tag in ("vehicle", "ship", "aircraft", "building"):
        indices = tag_to_indices.get(tag, [])
        if not indices:
            continue
        mask = np.isin(reduced_map, indices)
        counts[tag] = estimate_connected_components(mask, min_pixels=3 if tag in {"vehicle", "ship", "aircraft"} else 8)
    return counts


def fit_image_to_box(image: Image.Image, size: tuple[int, int], bg_color: str = "#0d1117") -> Image.Image:
    preview = ImageOps.contain(image.convert("RGB"), size, method=RESAMPLE_BILINEAR)
    canvas = Image.new("RGB", size, bg_color)
    x = (size[0] - preview.width) // 2
    y = (size[1] - preview.height) // 2
    canvas.paste(preview, (x, y))
    return canvas


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_spacing: int = 6,
) -> int:
    x, y = position
    current_y = y
    for paragraph in text.split("\n"):
        if not paragraph:
            current_y += getattr(font, "size", 16) + line_spacing
            continue
        line = ""
        for char in paragraph:
            candidate = line + char
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width or not line:
                line = candidate
            else:
                draw.text((x, current_y), line, font=font, fill=fill)
                current_y += (bbox[3] - bbox[1]) + line_spacing
                line = char
        if line:
            bbox = draw.textbbox((0, 0), line, font=font)
            draw.text((x, current_y), line, font=font, fill=fill)
            current_y += (bbox[3] - bbox[1]) + line_spacing
    return current_y


def draw_panel(canvas: Image.Image, box: tuple[int, int, int, int], title: str, image: Image.Image, subtitle: str = "") -> None:
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=24, fill="#151b24", outline="#2e3643", width=2)
    draw.text((x1 + 22, y1 + 16), title, font=get_font(26, bold=True), fill="#ffffff")
    if subtitle:
        draw.text((x1 + 22, y1 + 48), subtitle, font=get_font(14), fill="#9fb0c8")
    preview = fit_image_to_box(image, (x2 - x1 - 44, y2 - y1 - 90), bg_color="#0d1117")
    canvas.paste(preview, (x1 + 22, y1 + 74))


def create_stats_card(result: SegmentationResult, size: tuple[int, int] = (720, 420)) -> Image.Image:
    indicators = get_structural_indicators(result)
    dominant = dominant_name_for_poster(result)
    prompt_label = prompt_label_for_poster(result.prompt_request.label)

    card = Image.new("RGB", size, "#101c27")
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=28, fill="#101c27", outline="#294151", width=2)
    draw.text((26, 22), "Structured Metrics", font=get_font(28, bold=True), fill="#ffffff")
    draw.text(
        (26, 56),
        f"{result.dataset_name} | {prompt_label} | Latency {result.latency_s:.2f}s | Device {result.device}",
        font=get_font(15),
        fill="#9fb0c8",
    )

    metrics = [
        ("Built-up", indicators["built_up"], "#6fd7ff"),
        ("Blue-Green", indicators["blue_green"], "#88f1b6"),
        ("Bare Land", indicators["bareland"], "#ffd27b"),
        ("Small Targets", float(indicators["mobility_targets"]), "#ff8e7c"),
    ]
    for index, (label, value, color) in enumerate(metrics):
        left = 26 + index * 168
        draw.rounded_rectangle((left, 92, left + 148, 168), radius=18, fill="#0a141d", outline="#1f3443", width=2)
        draw.text((left + 14, 108), label, font=get_font(13, bold=True), fill=color)
        value_text = f"{int(value)}" if label == "Small Targets" else f"{value * 100:.1f}%"
        draw.text((left + 14, 132), value_text, font=get_font(21, bold=True), fill="#ffffff")

    draw.text((26, 198), f"Dominant Class: {dominant}", font=get_font(16, bold=True), fill="#dde6f3")
    for row_index, item in enumerate(result.distribution[:6]):
        top = 232 + row_index * 28
        color = item["color"]
        draw.rounded_rectangle((26, top, 230, top + 18), radius=8, fill="#0d1721")
        draw.rounded_rectangle((26, top, 44, top + 18), radius=8, fill=color)
        draw.text((52, top - 1), item["name"], font=get_font(14, bold=True), fill="#dde6f3")
        bar_left = 252
        bar_width = size[0] - bar_left - 36
        draw.rounded_rectangle((bar_left, top, bar_left + bar_width, top + 18), radius=8, fill="#1a2c39")
        fill_width = max(12, int(bar_width * float(item["ratio"])))
        draw.rounded_rectangle((bar_left, top, bar_left + fill_width, top + 18), radius=8, fill=color)
        draw.text((bar_left + bar_width - 92, top - 1), f"{float(item['ratio']) * 100:5.1f}%", font=get_font(14), fill="#9fb0c8")
    return card


def create_assessment_card(result: SegmentationResult, scenario_name: str, size: tuple[int, int] = (720, 420)) -> Image.Image:
    assessment = build_scenario_assessment_en(result, scenario_name)
    scenario_title = scenario_name_for_poster(scenario_name)

    card = Image.new("RGB", size, "#101c27")
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=28, fill="#101c27", outline="#294151", width=2)
    draw.text((26, 22), "Scene Assessment", font=get_font(28, bold=True), fill="#ffffff")
    draw.text((26, 58), scenario_title, font=get_font(17, bold=True), fill="#6fd7ff")
    draw.text((26, 84), f"Scene Fitness {assessment['score']} / 100 · {assessment['level']}", font=get_font(16, bold=True), fill="#ffffff")
    current_y = draw_wrapped_text(
        draw,
        str(assessment["headline"]),
        (26, 110),
        get_font(15),
        "#9fb0c8",
        size[0] - 52,
        line_spacing=4,
    )
    current_y += 10
    for bullet in assessment["bullets"][:4]:
        current_y = draw_wrapped_text(
            draw,
            f"• {bullet}",
            (30, current_y),
            get_font(15),
            "#dbe7ef",
            size[0] - 56,
            line_spacing=4,
        )
        current_y += 8
    draw.rounded_rectangle((26, size[1] - 104, size[0] - 26, size[1] - 26), radius=18, fill="#0b141d", outline="#1f3443", width=2)
    draw_wrapped_text(
        draw,
        f"Decision Hint: {scenario_goal_for_poster(scenario_name)}",
        (42, size[1] - 90),
        get_font(15),
        "#88f1b6",
        size[0] - 84,
        line_spacing=4,
    )
    return card


def create_single_poster(result: SegmentationResult, scenario_name: str) -> Image.Image:
    stats_card = create_stats_card(result)
    assessment_card = create_assessment_card(result, scenario_name)
    poster = Image.new("RGB", (1600, 1020), "#09101a")
    draw = ImageDraw.Draw(poster)
    draw.rounded_rectangle((18, 18, 1582, 1002), radius=30, fill="#0d1620", outline="#2a4050", width=2)
    draw.text((46, 40), POSTER_BRAND_NAME, font=get_font(34, bold=True), fill="#ffffff")
    subtitle = (
        f"Dataset: {result.dataset_name}   Prompt Set: {prompt_label_for_poster(result.prompt_request.label)}   "
        f"Scene: {scenario_name_for_poster(scenario_name)}   Classes: {result.prompt_request.class_count}"
    )
    draw.text((46, 84), subtitle, font=get_font(16), fill="#6fd7ff")
    draw_panel(poster, (40, 136, 780, 530), "Segmentation Overlay", result.overlay_image)
    draw_panel(poster, (820, 136, 1560, 530), "Semantic Mask", result.mask_image)
    draw_panel(poster, (40, 580, 780, 972), "Metrics Overview", stats_card)
    draw_panel(
        poster,
        (820, 580, 1560, 972),
        "Scene Assessment",
        assessment_card,
        subtitle=f"{scenario_name_for_poster(scenario_name)} · {prompt_label_for_poster(result.prompt_request.label)}",
    )
    return poster


def resolve_uploaded_image(file_obj: object) -> tuple[Image.Image, str]:
    if isinstance(file_obj, Image.Image):
        return file_obj.convert("RGB"), "uploaded_image"
    if isinstance(file_obj, (str, Path)):
        path = Path(file_obj)
        return Image.open(path).convert("RGB"), path.name
    if isinstance(file_obj, dict) and file_obj.get("name"):
        path = Path(str(file_obj["name"]))
        return Image.open(path).convert("RGB"), path.name
    if hasattr(file_obj, "name"):
        path = Path(str(getattr(file_obj, "name")))
        return Image.open(path).convert("RGB"), path.name
    raise ValueError("无法识别输入图像。")


def infer_segmentation(
    image: Image.Image,
    dataset_name: str,
    prompt_request: PromptRequest,
    device: str,
    transparency: float,
) -> SegmentationResult:
    session, cache_hit = get_model_session(prompt_request, device)
    pil_image = image.convert("RGB")
    start_time = time.perf_counter()
    with session.lock:
        seg_logits = session.model._inference_single_view(pil_image)
        class_logits = aggregate_query_logits_to_class_logits(seg_logits, session.model.query_idx, session.model.num_cls)
    latency_s = time.perf_counter() - start_time

    has_background_class = bool(prompt_request.class_names) and prompt_request.class_names[0].lower() in {"background", "bg"}
    if hasattr(session.model, "_build_prediction_outputs") and has_background_class:
        label_map_tensor, pred_instances = session.model._build_prediction_outputs(class_logits)
        label_map = label_map_tensor.detach().cpu().numpy().astype(np.int64)
    else:
        label_map = torch.argmax(class_logits, dim=0).detach().cpu().numpy().astype(np.int64)
        pred_instances = None

    palette = get_palette_rgb(dataset_name)
    distribution = summarize_distribution(label_map, prompt_request.class_names, palette)
    overlay_image, mask_image = blend_images(np.asarray(pil_image), colorize_prediction(label_map, palette), label_map, transparency)
    semantic_ratios = aggregate_distribution_by_semantic_tag(distribution)
    component_counts = estimate_component_counts(label_map, prompt_request.class_names)

    return SegmentationResult(
        dataset_name=dataset_name,
        prompt_request=prompt_request,
        device=normalize_device_choice(device),
        latency_s=latency_s,
        cache_hit=cache_hit,
        label_map=label_map,
        overlay_image=overlay_image,
        mask_image=mask_image,
        distribution=distribution,
        dominant_class=get_dominant_foreground_class(distribution),
        semantic_ratios=semantic_ratios,
        component_counts=component_counts,
        pred_instances=pred_instances,
    )


def format_distribution_rows(distribution: list[dict[str, object]], limit: int = 8) -> str:
    rows = []
    for item in distribution[:limit]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['name']))}</td>"
            f"<td>{float(item['ratio']) * 100:.2f}%</td>"
            f"<td style='font-family: var(--font-mono);'>{int(item['pixels'])}</td>"
            "</tr>"
        )
    return "".join(rows) if rows else "<tr><td colspan='3'>暂无分割统计</td></tr>"


def prompt_label_for_poster(label: str) -> str:
    return {
        "标准词表": "Standard Prompt Set",
        "增强词表": "Enhanced Prompt Set",
        "自定义词表": "Custom Prompt Set",
    }.get(label, label)


def scenario_name_for_poster(scenario_name: str) -> str:
    return {
        "城市与交通治理": "Urban and Traffic Governance",
        "环境与人类发展": "Environment and Human Development",
        "农业与资源监测": "Agriculture and Resource Monitoring",
        "综合巡检与应急保障": "Inspection and Emergency Support",
    }.get(scenario_name, scenario_name)


def scenario_goal_for_poster(scenario_name: str) -> str:
    return {
        "城市与交通治理": "Focus on roads, buildings and fine targets to support urban governance and traffic interpretation.",
        "环境与人类发展": "Focus on water, vegetation, cropland and built-up coupling to support environmental governance and settlement evaluation.",
        "农业与资源监测": "Focus on cropland, water, bare land and built-up disturbance to support resource survey and agricultural monitoring.",
        "综合巡检与应急保障": "Focus on built-up areas, disturbance and high-frequency small targets to support patrol, warning and emergency deployment.",
    }.get(scenario_name, scenario_name_for_poster(scenario_name))


def dominant_name_for_poster(result: SegmentationResult) -> str:
    if result.dominant_class is None:
        return "Background"
    name = str(result.dominant_class["name"]).strip()
    return "Background" if name in {"背景", "background", "bg"} else name


def get_structural_indicators(result: SegmentationResult) -> dict[str, float]:
    semantic = result.semantic_ratios
    vehicle_targets = float(result.component_counts.get("vehicle", 0))
    ship_targets = float(result.component_counts.get("ship", 0))
    aircraft_targets = float(result.component_counts.get("aircraft", 0))
    mobility_targets = vehicle_targets + ship_targets + aircraft_targets
    return {
        "building": semantic.get("building", 0.0),
        "road": semantic.get("road", 0.0),
        "water": semantic.get("water", 0.0),
        "vegetation": semantic.get("vegetation", 0.0),
        "agriculture": semantic.get("agriculture", 0.0),
        "bareland": semantic.get("bareland", 0.0),
        "built_up": semantic.get("building", 0.0) + semantic.get("road", 0.0),
        "blue_green": semantic.get("water", 0.0) + semantic.get("vegetation", 0.0) + semantic.get("agriculture", 0.0),
        "mobility_ratio": semantic.get("vehicle", 0.0) + semantic.get("ship", 0.0) + semantic.get("aircraft", 0.0),
        "vehicle_targets": vehicle_targets,
        "ship_targets": ship_targets,
        "aircraft_targets": aircraft_targets,
        "mobility_targets": mobility_targets,
    }


def infer_scene_profile(result: SegmentationResult) -> tuple[str, str]:
    indicators = get_structural_indicators(result)
    if indicators["road"] >= 0.15 and indicators["built_up"] >= 0.48:
        return "建成区道路骨架场景", "道路与建成地表占比较高，适合做城市空间结构与交通骨架观察。"
    if indicators["blue_green"] >= 0.46 and indicators["water"] >= 0.10:
        return "蓝绿生态复合场景", "水体与植被共存明显，更适合环境格局和生态覆盖分析。"
    if indicators["agriculture"] >= 0.24:
        return "农业资源利用场景", "农田覆盖较突出，适合农业生产与资源监测。"
    if indicators["bareland"] >= 0.18:
        return "开发扰动场景", "裸地比例偏高，区域可能存在建设扰动或待开发地块。"
    return "混合地表场景", "图像同时包含建设区与自然地表，适合做综合结构判读。"


def infer_scene_profile_en(result: SegmentationResult) -> tuple[str, str]:
    indicators = get_structural_indicators(result)
    if indicators["road"] >= 0.15 and indicators["built_up"] >= 0.48:
        return "Built-up road skeleton scene", "Road and built-up surfaces dominate the image, which is suitable for urban structure and traffic-corridor observation."
    if indicators["blue_green"] >= 0.46 and indicators["water"] >= 0.10:
        return "Blue-green ecological scene", "Water and vegetation co-exist clearly, which is suitable for ecological structure and environmental coverage analysis."
    if indicators["agriculture"] >= 0.24:
        return "Agricultural resource scene", "Cropland is prominent in the image, making it suitable for agricultural monitoring and resource observation."
    if indicators["bareland"] >= 0.18:
        return "Disturbance and development scene", "Bare land is relatively high and may indicate construction disturbance or undeveloped parcels."
    return "Mixed land-cover scene", "The image contains both built-up and natural surfaces, which is suitable for integrated structural interpretation."


def build_scenario_assessment(result: SegmentationResult, scenario_name: str) -> dict[str, object]:
    indicators = get_structural_indicators(result)
    dominant = result.dominant_class["name"] if result.dominant_class else "背景"
    profile_name, profile_note = infer_scene_profile(result)

    if scenario_name == "城市与交通治理":
        raw_score = 0.42 * indicators["built_up"] + 0.33 * indicators["road"] + 0.25 * min(indicators["mobility_targets"] / 12.0, 1.0)
        headline = "结果更适合支撑城市建设强度、路网结构和局部交通活跃度分析。"
        bullets = [
            "道路骨架率较高时，可直接支持道路连通性和建成区结构研判。"
            if indicators["road"] >= 0.12
            else "道路占比有限，当前结果更适合做地表结构识别，而不是精细交通流研判。",
            "建筑与道路共同占比较高，区域呈现明显建成环境特征。"
            if indicators["built_up"] >= 0.50
            else "建成区强度中等，区域仍保留一定生态或农业缓冲空间。",
            "检测到较多细粒目标，可辅助观察车辆等活动目标分布。"
            if indicators["mobility_targets"] >= 6
            else "细粒目标数量不高，城市分析应更多依赖道路与建成区形态。",
            f"主导地物为“{dominant}”，可作为当前地表组织的第一观察对象。",
        ]
    elif scenario_name == "环境与人类发展":
        raw_score = (
            0.40 * indicators["blue_green"]
            + 0.25 * indicators["water"]
            + 0.15 * indicators["vegetation"]
            + 0.20 * max(0.0, 1.0 - indicators["built_up"])
        )
        headline = "结果更适合用于蓝绿空间、人居环境与硬化地表平衡评估。"
        bullets = [
            "蓝绿空间占比较高，适合评估生态覆盖和环境承载能力。"
            if indicators["blue_green"] >= 0.35
            else "蓝绿空间占比较低，需要重点关注硬化地表扩张与生态挤压。",
            "水体信号较清晰，可进一步观察水陆边界与周边建设干扰。"
            if indicators["water"] >= 0.08
            else "水体占比不高，当前更偏向一般生态覆盖与人居结构分析。",
            "建成区压力相对可控，人居与生态空间存在一定缓冲。"
            if indicators["built_up"] <= 0.42
            else "建成区强度较高，需关注生态空间被压缩的风险。",
            f"当前地表画像接近“{profile_name}”，说明场景具有明显的人地关系特征。",
        ]
    elif scenario_name == "农业与资源监测":
        raw_score = (
            0.50 * indicators["agriculture"]
            + 0.20 * indicators["water"]
            + 0.10 * indicators["vegetation"]
            + 0.20 * max(0.0, 1.0 - indicators["built_up"])
        )
        headline = "结果更适合观察农田、水系支撑与资源利用扰动。"
        bullets = [
            "农田覆盖突出，可直接支撑耕地识别和农业空间统计。"
            if indicators["agriculture"] >= 0.22
            else "农田占比有限，当前图像不属于典型高强度农业场景。",
            "水体占比较高，有利于观察灌溉条件与资源配置边界。"
            if indicators["water"] >= 0.06
            else "水体信号不强，资源分析更应关注农田与裸地变化。",
            "裸地比例偏高，可能对应施工扰动、耕地轮作或资源开发影响。"
            if indicators["bareland"] >= 0.12
            else "裸地扰动可控，更适合做稳定资源格局分析。",
            f"主导地物为“{dominant}”，可作为资源利用结构的主要观察对象。",
        ]
    else:
        raw_score = (
            0.35 * indicators["built_up"]
            + 0.20 * indicators["bareland"]
            + 0.25 * min(indicators["mobility_targets"] / 10.0, 1.0)
            + 0.20 * max(indicators["water"], indicators["blue_green"] * 0.5)
        )
        headline = "结果适合做综合巡检、复杂地表排查和应急辅助观察。"
        bullets = [
            "细粒目标数量较多，可辅助巡检场景中的活跃目标排查。"
            if indicators["mobility_targets"] >= 5
            else "细粒目标不算密集，更适合做宏观结构巡检而非目标级告警。",
            "裸地扰动较明显，适合关注施工、翻耕或异常裸露区域。"
            if indicators["bareland"] >= 0.12
            else "裸地扰动较弱，当前画面整体更偏稳定地表。",
            "建成区率较高，可支持重点区域巡查和设施周边环境观察。"
            if indicators["built_up"] >= 0.45
            else "建成区并非绝对主导，巡检时需结合自然地表混合分布理解场景。",
            f"当前地表画像为“{profile_name}”，适合做快速场景归类与优先级判别。",
        ]

    score = int(round(max(0.0, min(raw_score, 1.0)) * 100))
    level = "高适配" if score >= 70 else "中适配" if score >= 45 else "低适配"
    return {
        "score": score,
        "level": level,
        "headline": headline,
        "bullets": bullets[:4],
        "profile_name": profile_name,
        "profile_note": profile_note,
    }


def build_scenario_assessment_en(result: SegmentationResult, scenario_name: str) -> dict[str, object]:
    indicators = get_structural_indicators(result)
    dominant = dominant_name_for_poster(result)
    profile_name, profile_note = infer_scene_profile_en(result)

    if scenario_name == "城市与交通治理":
        raw_score = 0.42 * indicators["built_up"] + 0.33 * indicators["road"] + 0.25 * min(indicators["mobility_targets"] / 12.0, 1.0)
        headline = "This result is suitable for urban intensity, road skeleton and local traffic-activity analysis."
        bullets = [
            "A strong road skeleton can support connectivity inspection and built-up structure interpretation."
            if indicators["road"] >= 0.12
            else "Road coverage is limited, so the current result is better for general land-cover reading than fine traffic-flow analysis.",
            "Buildings and roads jointly occupy a large portion of the image, showing a clear urbanized pattern."
            if indicators["built_up"] >= 0.50
            else "Built-up intensity is moderate and the area still preserves ecological or agricultural buffer space.",
            "Multiple small targets are detected and can support observation of vehicles and other active objects."
            if indicators["mobility_targets"] >= 6
            else "Small-target density is not high, so urban analysis should rely more on road and built-up morphology.",
            f"The dominant land-cover class is '{dominant}', which is the strongest spatial signal in the current output.",
        ]
    elif scenario_name == "环境与人类发展":
        raw_score = (
            0.40 * indicators["blue_green"]
            + 0.25 * indicators["water"]
            + 0.15 * indicators["vegetation"]
            + 0.20 * max(0.0, 1.0 - indicators["built_up"])
        )
        headline = "This result is suitable for blue-green space, settlement quality and hardened-surface balance assessment."
        bullets = [
            "Blue-green space is prominent and supports ecological coverage and environmental carrying-capacity analysis."
            if indicators["blue_green"] >= 0.35
            else "Blue-green space is limited, so attention should be paid to ecological compression by hardened surfaces.",
            "Water coverage is clear enough to support shoreline and surrounding disturbance observation."
            if indicators["water"] >= 0.08
            else "Water is not dominant, so the image is more suitable for general environmental-structure assessment.",
            "Built-up pressure is relatively controllable and there is still some buffer between settlement and ecological space."
            if indicators["built_up"] <= 0.42
            else "Built-up intensity is high and ecological space compression should be treated as a risk.",
            f"The current scene profile is '{profile_name}', indicating a clear human-environment coupling pattern.",
        ]
    elif scenario_name == "农业与资源监测":
        raw_score = (
            0.50 * indicators["agriculture"]
            + 0.20 * indicators["water"]
            + 0.10 * indicators["vegetation"]
            + 0.20 * max(0.0, 1.0 - indicators["built_up"])
        )
        headline = "This result is suitable for cropland, water-support and resource-disturbance monitoring."
        bullets = [
            "Cropland coverage is prominent and directly supports agricultural-space statistics."
            if indicators["agriculture"] >= 0.22
            else "Cropland share is limited, so the image is not a strongly agricultural scene.",
            "Water coverage is meaningful and helps interpret irrigation support and resource-allocation boundaries."
            if indicators["water"] >= 0.06
            else "Water signal is weak, so resource interpretation should focus more on cropland and bare-land changes.",
            "Bare land is relatively high and may indicate disturbance, rotation or resource-development impact."
            if indicators["bareland"] >= 0.12
            else "Bare-land disturbance is limited and the scene is more suitable for stable resource-pattern analysis.",
            f"The dominant land-cover class is '{dominant}', which should be treated as the main resource-structure signal.",
        ]
    else:
        raw_score = (
            0.35 * indicators["built_up"]
            + 0.20 * indicators["bareland"]
            + 0.25 * min(indicators["mobility_targets"] / 10.0, 1.0)
            + 0.20 * max(indicators["water"], indicators["blue_green"] * 0.5)
        )
        headline = "This result is suitable for integrated inspection, disturbance screening and emergency-support observation."
        bullets = [
            "Small-target density is relatively high and can support active-object inspection."
            if indicators["mobility_targets"] >= 5
            else "Small-target density is not high, so the scene is more suitable for macro inspection than alert-level monitoring.",
            "Bare-land disturbance is visible and suitable for construction or abnormal exposure screening."
            if indicators["bareland"] >= 0.12
            else "Bare-land disturbance is limited and the current area appears relatively stable.",
            "Built-up intensity is high enough to support key-area patrol and surrounding-facility observation."
            if indicators["built_up"] >= 0.45
            else "Built-up areas are not fully dominant, so inspection should consider mixed natural and built surfaces.",
            f"The current scene profile is '{profile_name}', which is useful for fast scene tagging and priority judgment.",
        ]

    score = int(round(max(0.0, min(raw_score, 1.0)) * 100))
    level = "High Fit" if score >= 70 else "Medium Fit" if score >= 45 else "Low Fit"
    return {
        "score": score,
        "level": level,
        "headline": headline,
        "bullets": bullets[:4],
        "profile_name": profile_name,
        "profile_note": profile_note,
    }


def build_structure_metrics(result: SegmentationResult, scenario_name: str) -> list[tuple[str, str, str]]:
    indicators = get_structural_indicators(result)
    assessment = build_scenario_assessment(result, scenario_name)
    dominant = result.dominant_class["name"] if result.dominant_class else "N/A"

    if scenario_name == "城市与交通治理":
        return [
            ("场景适配度", f"{assessment['score']} / 100", str(assessment["level"])),
            ("道路骨架率", f"{indicators['road'] * 100:.2f}%", "道路相关地表占比"),
            ("建成区强度", f"{indicators['built_up'] * 100:.2f}%", "建筑 + 道路"),
            ("细粒目标数", str(int(indicators["mobility_targets"])), f"主导地物：{dominant}"),
        ]
    if scenario_name == "环境与人类发展":
        return [
            ("场景适配度", f"{assessment['score']} / 100", str(assessment["level"])),
            ("蓝绿空间率", f"{indicators['blue_green'] * 100:.2f}%", "水体 + 植被 + 农田"),
            ("水体占比", f"{indicators['water'] * 100:.2f}%", "关键生态要素"),
            ("硬化强度", f"{indicators['built_up'] * 100:.2f}%", f"主导地物：{dominant}"),
        ]
    if scenario_name == "农业与资源监测":
        return [
            ("场景适配度", f"{assessment['score']} / 100", str(assessment["level"])),
            ("农田占比", f"{indicators['agriculture'] * 100:.2f}%", "农业资源主体"),
            ("水体支撑", f"{indicators['water'] * 100:.2f}%", "资源配置相关"),
            ("裸地扰动", f"{indicators['bareland'] * 100:.2f}%", f"主导地物：{dominant}"),
        ]
    return [
        ("场景适配度", f"{assessment['score']} / 100", str(assessment["level"])),
        ("建成区率", f"{indicators['built_up'] * 100:.2f}%", "重点区域强度"),
        ("裸地扰动", f"{indicators['bareland'] * 100:.2f}%", "异常地表线索"),
        ("细粒目标数", str(int(indicators["mobility_targets"])), f"主导地物：{dominant}"),
    ]


def build_structural_summary(result: SegmentationResult, scenario_name: str) -> list[str]:
    assessment = build_scenario_assessment(result, scenario_name)
    dominant = result.dominant_class["name"] if result.dominant_class else "背景"
    return [
        f"地表画像：{assessment['profile_name']}。{assessment['profile_note']}",
        *[str(text) for text in assessment["bullets"][:3]],
        f"当前主导地物为“{dominant}”，说明该类别是本次分割输出中最强的空间信号。",
    ]


def build_experiment_basis_html(dataset_name: str) -> str:
    if dataset_name == "Vaihingen":
        base_values = EXPERIMENT_SNAPSHOT.get("Vaihingen_base", {})
        exp_values = EXPERIMENT_SNAPSHOT.get("Vaihingen_expanded", {})
        if base_values and exp_values:
            gain_miou = exp_values.get("mIoU", 0.0) - base_values.get("mIoU", 0.0)
            return (
                "<div class='section-copy'>"
                f"离线实验显示，Vaihingen 数据集上增强词表将最佳 mIoU 从 {_metric_text(base_values, 'mIoU')} "
                f"提升到 {_metric_text(exp_values, 'mIoU')}，提升 {gain_miou:.2f}，这为当前建成区边界识别提供了更稳定的词表基础。"
                "</div>"
            )
    elif dataset_name in EXPERIMENT_SNAPSHOT and EXPERIMENT_SNAPSHOT[dataset_name]:
        values = EXPERIMENT_SNAPSHOT[dataset_name]
        return (
            "<div class='section-copy'>"
            f"当前数据集离线实验最佳 mIoU 为 {_metric_text(values, 'mIoU')}，"
            f"AP50 为 {_metric_text(values, 'AP50')}，可作为本次可视化分析的性能背景。"
            "</div>"
        )
    return "<div class='section-copy'>当前没有找到与该数据集对应的离线实验记录。</div>"


def build_qwen_context(dataset_name: str, result: SegmentationResult, scenario_name: str) -> str:
    indicators = get_structural_indicators(result)
    assessment = build_scenario_assessment(result, scenario_name)
    scenario = SCENARIO_SPECS.get(scenario_name, SCENARIO_SPECS[DEFAULT_SCENARIO])
    dominant = result.dominant_class["name"] if result.dominant_class else "背景"
    top_lines = ", ".join(f"{item['name']}:{float(item['ratio']) * 100:.2f}%" for item in result.distribution[:6])
    context = [
        f"数据集: {dataset_name}",
        f"评估场景: {scenario.name}",
        f"场景说明: {scenario.summary}",
        f"决策目标: {scenario.decision_goal}",
        f"关注重点: {scenario.focus_hint}",
        f"分割方案: {result.prompt_request.label}",
        f"地表画像: {assessment['profile_name']}",
        f"场景适配度: {assessment['score']}/100 ({assessment['level']})",
        f"主导地物: {dominant}",
        f"面积分布: {top_lines}",
        f"建成区率: {indicators['built_up'] * 100:.2f}%",
        f"道路占比: {indicators['road'] * 100:.2f}%",
        f"蓝绿空间率: {indicators['blue_green'] * 100:.2f}%",
        f"水体占比: {indicators['water'] * 100:.2f}%",
        f"农田占比: {indicators['agriculture'] * 100:.2f}%",
        f"裸地占比: {indicators['bareland'] * 100:.2f}%",
        f"细粒目标数量: {int(indicators['mobility_targets'])}",
    ]
    context.extend(f"规则判断: {bullet}" for bullet in assessment["bullets"][:4])

    if dataset_name == "Vaihingen":
        base_values = EXPERIMENT_SNAPSHOT.get("Vaihingen_base", {})
        exp_values = EXPERIMENT_SNAPSHOT.get("Vaihingen_expanded", {})
        if base_values and exp_values:
            context.append(
                f"离线实验依据: 标准词表最佳 mIoU 为 {_metric_text(base_values, 'mIoU')}，增强词表最佳 mIoU 为 {_metric_text(exp_values, 'mIoU')}。"
            )
    elif dataset_name in EXPERIMENT_SNAPSHOT and EXPERIMENT_SNAPSHOT[dataset_name]:
        values = EXPERIMENT_SNAPSHOT[dataset_name]
        context.append(f"离线实验依据: 当前最佳 mIoU 为 {_metric_text(values, 'mIoU')}，AP50 为 {_metric_text(values, 'AP50')}。")

    return "\n".join(context)


def build_overview_html(
    dataset_name: str,
    main_result: SegmentationResult,
    scenario_name: str,
    sam_device: str,
    qwen_device: str,
    enable_qwen_analysis: bool,
) -> str:
    scenario = SCENARIO_SPECS.get(scenario_name, SCENARIO_SPECS[DEFAULT_SCENARIO])
    assessment = build_scenario_assessment(main_result, scenario_name)
    structure_metrics = render_metrics_row(build_structure_metrics(main_result, scenario_name))
    structural_summary = "".join(
        f"<div class='insight-item'>{html.escape(text)}</div>" for text in build_structural_summary(main_result, scenario_name)
    )

    if enable_qwen_analysis and sam_device == qwen_device and sam_device.startswith("cuda"):
        qwen_note = (
            "<div class='status-box warn' style='margin-top: 16px;'>"
            "<div class='status-title'>设备提醒</div>"
            "<div class='status-copy'>"
            "当前 Qwen 与 SAM3 使用同一块 GPU。为了降低显存占用风险，建议将 Qwen 切换到另一块 GPU 后再执行场景评估。"
            "</div></div>"
        )
    elif enable_qwen_analysis:
        qwen_note = (
            "<div class='status-box' style='margin-top: 16px;'>"
            "<div class='status-title'>Qwen 场景评估已启用</div>"
            f"<div class='status-copy'>Qwen 设备：{html.escape(qwen_device)}。系统将围绕“{html.escape(scenario.name)}”输出业务解读与辅助决策建议。</div>"
            "</div>"
        )
    else:
        qwen_note = (
            "<div class='status-box' style='margin-top: 16px;'>"
            "<div class='status-title'>当前为结构化展示模式</div>"
            "<div class='status-copy'>尚未启用 Qwen 场景评估，页面当前展示的是遥感分割结果与规则化结构指标。</div>"
            "</div>"
        )

    return (
        "<div class='glass-card section-shell'>"
        "<div class='section-title'>场景化结果解读</div>"
        f"<div class='section-copy'><strong>{html.escape(scenario.name)}</strong>：{html.escape(scenario.summary)}</div>"
        f"<div class='section-copy'><strong>场景结论：</strong>{html.escape(str(assessment['headline']))}</div>"
        f"{build_experiment_basis_html(dataset_name)}"
        f"{structure_metrics}"
        "<div class='section-copy' style='margin-top: 18px; margin-bottom: 8px;'>关键发现</div>"
        f"<div class='insight-list'>{structural_summary}</div>"
        "<div class='section-copy' style='margin-top: 18px; margin-bottom: 6px;'>地物分布统计</div>"
        "<table class='result-table'>"
        "<thead><tr><th>类别</th><th>面积占比</th><th>像素数</th></tr></thead>"
        f"<tbody>{format_distribution_rows(main_result.distribution)}</tbody>"
        "</table>"
        f"{qwen_note}"
        "</div>"
    )


def build_markdown_placeholder(text: str) -> str:
    return text


def analyze_with_qwen(
    image: Image.Image,
    dataset_name: str,
    result: SegmentationResult,
    scenario_name: str,
    qwen_device: str,
    sam_device: str,
) -> str:
    if qwen_device == sam_device and qwen_device.startswith("cuda"):
        return "### Qwen 场景评估未执行\n\nQwen 与 SAM3 当前配置为同一块 GPU。为避免显存溢出，请将 Qwen 切换到另一块 GPU。"

    session, _ = get_qwen_session(dataset_name, qwen_device)
    context_text = build_qwen_context(dataset_name, result, scenario_name)
    with session.lock:
        analysis = session.agent.analyze_remote_sensing_with_context(image, context_text, language="zh")
    if not analysis:
        return "### Qwen 场景评估未返回内容\n\n请检查 Qwen 模型加载状态、设备分配和图像输入。"
    return analysis


def run_single_analysis(
    image: Image.Image | None,
    dataset_name: str,
    scenario_name: str,
    prompt_mode: str,
    custom_prompts: str,
    sam_device: str,
    transparency: float,
    enable_qwen_analysis: bool,
    qwen_device: str,
) -> tuple[list[tuple[Image.Image, str]], str, str]:
    if image is None:
        return [], build_dataset_overview_html(dataset_name), build_markdown_placeholder("### 等待输入\n\n请先上传图像。")

    dataset_name = dataset_name if dataset_name in DATASET_SPECS else DEFAULT_DATASET
    scenario_name = scenario_name if scenario_name in SCENARIO_SPECS else DEFAULT_SCENARIO
    sam_device = normalize_device_choice(sam_device)
    qwen_device = normalize_device_choice(qwen_device)

    main_request = build_prompt_request(dataset_name, prompt_mode, custom_prompts)
    main_result = infer_segmentation(image, dataset_name, main_request, sam_device, transparency)
    assessment_card = create_assessment_card(main_result, scenario_name)
    gallery_items: list[tuple[Image.Image, str]] = [
        (create_single_poster(main_result, scenario_name), "成果海报"),
        (main_result.overlay_image, "分割叠加结果"),
        (main_result.mask_image, "语义分割图"),
        (create_stats_card(main_result), "结构化指标卡"),
        (assessment_card, "场景评估卡"),
    ]

    qwen_markdown = build_markdown_placeholder(
        "### Qwen 场景评估未启用\n\n勾选“启用 Qwen 场景评估”后，系统会围绕所选场景输出业务解读和辅助决策建议。"
    )

    if enable_qwen_analysis:
        qwen_markdown = analyze_with_qwen(image, dataset_name, main_result, scenario_name, qwen_device, sam_device)

    overview_html = build_overview_html(
        dataset_name=dataset_name,
        main_result=main_result,
        scenario_name=scenario_name,
        sam_device=sam_device,
        qwen_device=qwen_device,
        enable_qwen_analysis=enable_qwen_analysis,
    )
    return gallery_items, overview_html, qwen_markdown


def run_batch_analysis(
    files: list[object] | None,
    dataset_name: str,
    scenario_name: str,
    prompt_mode: str,
    custom_prompts: str,
    sam_device: str,
    transparency: float,
) -> tuple[list[tuple[Image.Image, str]], str]:
    if not files:
        return [], "<div class='status-box'><div class='status-title'>等待批量输入</div><div class='status-copy'>请上传多张图像后再执行批量分析。</div></div>"

    dataset_name = dataset_name if dataset_name in DATASET_SPECS else DEFAULT_DATASET
    scenario_name = scenario_name if scenario_name in SCENARIO_SPECS else DEFAULT_SCENARIO
    sam_device = normalize_device_choice(sam_device)
    prompt_request = build_prompt_request(dataset_name, prompt_mode, custom_prompts)
    gallery_items: list[tuple[Image.Image, str]] = []
    rows: list[dict[str, object]] = []

    for file_obj in files:
        try:
            image, file_name = resolve_uploaded_image(file_obj)
            result = infer_segmentation(image, dataset_name, prompt_request, sam_device, transparency)
            dominant = result.dominant_class["name"] if result.dominant_class else "N/A"
            assessment = build_scenario_assessment(result, scenario_name)
            gallery_items.append((result.overlay_image, f"{file_name} | {assessment['score']}/100 | {dominant}"))
            rows.append(
                {
                    "ok": True,
                    "file_name": file_name,
                    "latency_s": result.latency_s,
                    "dominant": dominant,
                    "built_up": get_structural_indicators(result)["built_up"],
                    "blue_green": get_structural_indicators(result)["blue_green"],
                    "score": assessment["score"],
                    "profile": assessment["profile_name"],
                }
            )
        except Exception as exc:
            file_name = Path(str(getattr(file_obj, "name", "unknown_file"))).name
            gallery_items.append((Image.new("RGB", (600, 400), "#2a1212"), f"{file_name} | 失败"))
            rows.append(
                {
                    "ok": False,
                    "file_name": file_name,
                    "latency_s": 0.0,
                    "dominant": "-",
                    "built_up": 0.0,
                    "blue_green": 0.0,
                    "score": 0.0,
                    "profile": "-",
                    "error": str(exc),
                }
            )

    success_rows = [row for row in rows if row.get("ok")]
    avg_latency = sum(float(row["latency_s"]) for row in success_rows) / len(success_rows) if success_rows else 0.0
    avg_built_up = sum(float(row["built_up"]) for row in success_rows) / len(success_rows) if success_rows else 0.0
    avg_blue_green = sum(float(row["blue_green"]) for row in success_rows) / len(success_rows) if success_rows else 0.0
    avg_score = sum(float(row["score"]) for row in success_rows) / len(success_rows) if success_rows else 0.0
    dominant_counter = Counter(str(row["dominant"]) for row in success_rows)
    dominant_text = dominant_counter.most_common(1)[0][0] if dominant_counter else "N/A"
    profile_counter = Counter(str(row["profile"]) for row in success_rows)
    profile_text = profile_counter.most_common(1)[0][0] if profile_counter else "N/A"

    batch_metrics = render_metrics_row(
        [
            ("成功样本数", str(len(success_rows)), "有效图像"),
            ("平均时延", f"{avg_latency:.2f}s", "单张图像"),
            ("平均场景适配度", f"{avg_score:.1f} / 100", html.escape(scenario_name)),
            ("平均建成区 / 蓝绿", f"{avg_built_up * 100:.1f}% / {avg_blue_green * 100:.1f}%", "结构对比"),
        ]
    )

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['file_name']))}</td>"
            f"<td>{'成功' if row.get('ok') else '失败'}</td>"
            f"<td>{float(row.get('score', 0.0)):.0f}</td>"
            f"<td>{html.escape(str(row.get('dominant', '-')))}</td>"
            f"<td style='font-family: var(--font-mono);'>{float(row.get('latency_s', 0.0)):.2f}s</td>"
            "</tr>"
        )
    batch_html = (
        "<div class='glass-card section-shell'>"
        "<div class='section-title'>批量分析汇总</div>"
        f"<div class='section-copy'>当前批量任务使用“{html.escape(prompt_request.label)}”进行分割，并围绕“{html.escape(scenario_name)}”做场景评估。高频主导地物为“{html.escape(dominant_text)}”，主流地表画像为“{html.escape(profile_text)}”。</div>"
        f"{batch_metrics}"
        "<table class='result-table'>"
        "<thead><tr><th>文件名</th><th>状态</th><th>适配度</th><th>主导地物</th><th>时延</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody>"
        "</table>"
        "</div>"
    )
    return gallery_items, batch_html


def build_empty_overview() -> str:
    return "<div class='status-box'><div class='status-title'>等待分析</div><div class='status-copy'>请上传图像，选择分割词表与评估场景后执行分析。</div></div>"


def build_empty_batch() -> str:
    return "<div class='status-box'><div class='status-title'>等待批量任务</div><div class='status-copy'>请上传多张图像以生成批量统计结果。</div></div>"


def clear_outputs() -> tuple[list[tuple[Image.Image, str]], str, str, list[tuple[Image.Image, str]], str]:
    return [], build_empty_overview(), build_markdown_placeholder("### Qwen 场景评估\n\n未启用。"), [], build_empty_batch()


def release_cache_feedback() -> tuple[str, str]:
    cleared = clear_model_cache()
    message = (
        "<div class='status-box'>"
        "<div class='status-title'>缓存已释放</div>"
        f"<div class='status-copy'>已清理 {cleared} 个模型会话。下一次推理会重新加载缓存。</div>"
        "</div>"
    )
    return message, message


def build_example_rows() -> list[list[object]]:
    rows: list[list[object]] = []
    for dataset_name in get_available_datasets():
        for sample_path in list_sample_images(dataset_name)[:2]:
            rows.append(
                [
                    sample_path,
                    dataset_name,
                    DEFAULT_SCENARIO,
                    "增强词表",
                    "",
                    get_default_sam_device(),
                    0.55,
                    False,
                    get_default_qwen_device(),
                ]
            )
    return rows


def update_side_panels(
    dataset_name: str,
    scenario_name: str,
    prompt_mode: str,
    custom_prompts: str,
    enable_qwen_analysis: bool,
) -> tuple[str, str]:
    dataset_name = dataset_name if dataset_name in DATASET_SPECS else DEFAULT_DATASET
    scenario_name = scenario_name if scenario_name in SCENARIO_SPECS else DEFAULT_SCENARIO
    return (
        build_dataset_overview_html(dataset_name),
        build_prompt_preview_text(dataset_name, prompt_mode, custom_prompts, enable_qwen_analysis, scenario_name),
    )


def build_demo() -> gr.Blocks:
    example_rows = build_example_rows()
    with gr.Blocks(title=SYSTEM_NAME_FULL, css=APP_CSS, theme=gr.themes.Base()) as demo:
        gr.HTML(build_header_html())

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                dataset_info = gr.HTML(build_dataset_overview_html(DEFAULT_DATASET))
                with gr.Group(elem_classes=["glass-card", "section-shell"]):
                    gr.Markdown("### 参数设置")
                    dataset_choice = gr.Dropdown(choices=get_available_datasets(), value=DEFAULT_DATASET, label="数据集")
                    scenario_choice = gr.Dropdown(choices=get_available_scenarios(), value=DEFAULT_SCENARIO, label="评估场景")
                    prompt_mode_choice = gr.Radio(choices=["标准词表", "增强词表"], value="增强词表", label="分割词表方案")
                    custom_prompts_input = gr.Textbox(
                        label="自定义分割词表（可选）",
                        lines=2,
                        placeholder="留空时使用上方方案；如需自定义，可输入 building|house, road, water",
                    )
                    sam_device_choice = gr.Dropdown(choices=get_available_devices(), value=get_default_sam_device(), label="SAM3 设备")
                    transparency_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.55, step=0.01, label="叠加透明度", info="0 为原图，1 为纯分割图")
                    enable_qwen_checkbox = gr.Checkbox(label="启用 Qwen 场景评估", value=False)
                    qwen_device_choice = gr.Dropdown(
                        choices=get_available_devices(),
                        value=get_default_qwen_device(),
                        label="Qwen 设备",
                        info="建议与 SAM3 使用不同 GPU",
                    )
                    prompt_preview = gr.Textbox(
                        label="分割与评估预览",
                        value=build_prompt_preview_text(DEFAULT_DATASET, "增强词表", "", False, DEFAULT_SCENARIO),
                        lines=10,
                        interactive=False,
                    )
                    with gr.Row():
                        run_single_btn = gr.Button("执行单图分析", variant="primary", scale=2)
                        clear_btn = gr.Button("清空结果", variant="secondary", scale=1)
                    release_cache_btn = gr.Button("释放缓存", variant="secondary")

            with gr.Column(scale=7):
                with gr.Tabs():
                    with gr.Tab("单图分析"):
                        image_input = gr.Image(type="pil", label="上传图像")
                        if example_rows:
                            gr.Examples(
                                examples=example_rows,
                                inputs=[
                                    image_input,
                                    dataset_choice,
                                    scenario_choice,
                                    prompt_mode_choice,
                                    custom_prompts_input,
                                    sam_device_choice,
                                    transparency_slider,
                                    enable_qwen_checkbox,
                                    qwen_device_choice,
                                ],
                                label="示例图像",
                            )
                        single_gallery = gr.Gallery(label="结果图像", columns=2, object_fit="contain", height="auto")
                        single_overview = gr.HTML(build_empty_overview())
                        single_qwen = gr.Markdown(build_markdown_placeholder("### Qwen 场景评估\n\n未启用。"))

                    with gr.Tab("批量分析"):
                        multi_image_input = gr.File(file_count="multiple", file_types=["image"], label="上传多张图像")
                        run_batch_btn = gr.Button("执行批量分析", variant="primary")
                        batch_gallery = gr.Gallery(label="批量结果", columns=2, object_fit="contain", height="auto")
                        batch_overview = gr.HTML(build_empty_batch())

        for trigger in (
            dataset_choice.change,
            scenario_choice.change,
            prompt_mode_choice.change,
            custom_prompts_input.change,
            enable_qwen_checkbox.change,
        ):
            trigger(
                fn=update_side_panels,
                inputs=[dataset_choice, scenario_choice, prompt_mode_choice, custom_prompts_input, enable_qwen_checkbox],
                outputs=[dataset_info, prompt_preview],
            )

        run_single_btn.click(
            fn=run_single_analysis,
            inputs=[
                image_input,
                dataset_choice,
                scenario_choice,
                prompt_mode_choice,
                custom_prompts_input,
                sam_device_choice,
                transparency_slider,
                enable_qwen_checkbox,
                qwen_device_choice,
            ],
            outputs=[single_gallery, single_overview, single_qwen],
        )
        run_batch_btn.click(
            fn=run_batch_analysis,
            inputs=[
                multi_image_input,
                dataset_choice,
                scenario_choice,
                prompt_mode_choice,
                custom_prompts_input,
                sam_device_choice,
                transparency_slider,
            ],
            outputs=[batch_gallery, batch_overview],
        )
        clear_btn.click(
            fn=clear_outputs,
            outputs=[single_gallery, single_overview, single_qwen, batch_gallery, batch_overview],
        )
        release_cache_btn.click(
            fn=release_cache_feedback,
            outputs=[single_overview, batch_overview],
        )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port, share=False)
