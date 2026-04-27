import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
import sys
import tempfile
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from sam3_segmentor import SegEarthOV3Segmentation
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from mmengine.structures import BaseDataElement

# 导入palettes模块以获取官方配色方案
from palettes import _DATASET_METAINFO

# 导入QwenAgent用于智能解译
from core.qwen_agent import QwenAgent
from config import Config

def get_class_names_from_file(filepath):
    """从文件中读取类别名称"""
    if not os.path.exists(filepath):
        return ["background", "object"]
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    class_names = []
    for line in lines:
        line = line.strip()
        if ',' in line:
            # 解析逗号分隔的类别变体
            parts = [part.strip() for part in line.split(',')]
            class_names.extend(parts)
        else:
            class_names.append(line)
    
    return class_names


def get_base_class_mapping(filepath):
    """获取基础类别到索引的映射关系"""
    if not os.path.exists(filepath):
        return {}, []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    class_to_idx = {}
    idx_counter = 0
    for line in lines:
        line = line.strip()
        if ',' in line:
            # 解析逗号分隔的类别变体
            parts = [part.strip() for part in line.split(',')]
            for part in parts:
                class_to_idx[part] = idx_counter
        else:
            class_to_idx[line] = idx_counter
        idx_counter += 1
    
    return class_to_idx, list(class_to_idx.keys())


def merge_segmentation_results(seg_logits, query_idx, num_cls):
    """合并属于同一类别的不同提示词的分割结果，与sam3_segmentor.py保持一致"""
    # 如果类别数与查询数不一致，需要进行聚合
    if num_cls != seg_logits.shape[0]:
        seg_logits = seg_logits.unsqueeze(0)
        cls_index = torch.nn.functional.one_hot(query_idx)
        cls_index = cls_index.T.view(num_cls, len(query_idx), 1, 1).to(seg_logits.device)
        aggregated_logits = (seg_logits * cls_index).max(1)[0]
        return torch.argmax(aggregated_logits, dim=0).cpu().numpy()
    else:
        return torch.argmax(seg_logits, dim=0).cpu().numpy()


def aggregate_predictions_by_class(seg_logits, query_idx, num_cls):
    """
    根据类别索引聚合预测结果，与sam3_segmentor.py中的逻辑保持一致
    """
    # 如果类别数与查询数不一致，需要进行聚合
    if num_cls != seg_logits.shape[0]:
        seg_logits = seg_logits.unsqueeze(0)
        cls_index = torch.nn.functional.one_hot(query_idx)
        cls_index = cls_index.T.view(num_cls, len(query_idx), 1, 1).to(seg_logits.device)
        aggregated_logits = (seg_logits * cls_index).max(1)[0]
        # 只聚合logits，不直接argmax，与sam3_segmentor.py保持一致
        return aggregated_logits.cpu().numpy()
    else:
        return seg_logits.cpu().numpy()


def direct_segment_image(pil_image, query_words, device, dataset_type, transparency=0.5, use_expanded_prompts=False):
    # 创建临时的类名文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for cls in query_words:
            f.write(cls + "\n")
        temp_class_file = f.name
    
    # 根据数据集类型确定基础类别文件
    if dataset_type == "iSAID":
        base_class_file = "./configs/cls_iSAID.txt"
    elif dataset_type == "LoveDA":
        base_class_file = "./configs/cls_loveda.txt"
    elif dataset_type == "Potsdam":
        base_class_file = "./configs/cls_potsdam.txt"
    elif dataset_type == "Vaihingen":
        base_class_file = "./configs/cls_vaihingen.txt"
    else:
        base_class_file = "./configs/cls_iSAID.txt"  # 默认
    
    # 确定扩展提示池路径
    expanded_prompt_pool_path = None
    if use_expanded_prompts:
        # 根据数据集类型确定扩展提示词文件
        if dataset_type == "iSAID":
            exp_class_file = "./configs/cls_iSAID_exp.txt"
        elif dataset_type == "LoveDA":
            exp_class_file = "./configs/cls_loveda_exp.txt"
        elif dataset_type == "Potsdam":
            exp_class_file = "./configs/cls_potsdam_exp.txt"
        elif dataset_type == "Vaihingen":
            exp_class_file = "./configs/cls_vaihingen_exp.txt"
        else:
            exp_class_file = "./configs/cls_iSAID_exp.txt"  # 默认
        
        # 创建临时的扩展提示池pickle文件
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            temp_pkl_path = tmp_file.name
            
            # 从文本文件创建扩展提示池
            expanded_prompt_pool = {}
            with open(exp_class_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析形如 "building,house,structure" 的行
                    parts = [part.strip() for part in line.split(',') if part.strip()]
                    if len(parts) > 0:
                        # 使用整行内容作为键，各个部分作为该类的扩展提示词
                        key = ','.join(parts)  # 使用整行作为键
                        expanded_prompt_pool[key] = parts  # 所有部分都是该类的提示词变体
            
            # 保存为pickle文件
            import pickle
            with open(temp_pkl_path, 'wb') as f:
                pickle.dump(expanded_prompt_pool, f)
            
            expanded_prompt_pool_path = temp_pkl_path
    
    try:
        # 创建模型实例
        model = SegEarthOV3Segmentation(
            classname_path=temp_class_file,
            device=device,
            prob_thd=0.1,
            confidence_threshold=0.4,
            use_sem_seg=True,
            use_presence_score=True,
            use_transformer_decoder=True,
            expanded_prompt_pool_path=expanded_prompt_pool_path  # 正确设置扩展提示池路径
        )
        
        # 直接调用分割方法
        seg_logits = model._inference_single_view(pil_image)
        
        # 根据模型的类别索引聚合预测结果，与sam3_segmentor.py保持一致
        aggregated_logits = aggregate_predictions_by_class(seg_logits, model.query_idx, model.num_cls)
        
        # 应用与sam3_segmentor.py中predict方法相同的逻辑
        if model.num_cls != seg_logits.shape[0]:
            # 聚合后的logits进行argmax得到最终预测
            seg_pred = torch.from_numpy(aggregated_logits).argmax(0).cpu().numpy()
        else:
            # 没有聚合的情况
            seg_pred = torch.from_numpy(aggregated_logits).argmax(0).cpu().numpy()
        
        # 获取数据集的官方配色方案 (RGB格式)
        if dataset_type == "iSAID":
            palette_rgb = _DATASET_METAINFO['iSAIDDataset']['palette']
        elif dataset_type == "LoveDA":
            palette_rgb = _DATASET_METAINFO['LoveDADataset']['palette']
        elif dataset_type == "Potsdam":
            palette_rgb = _DATASET_METAINFO['PotsdamDataset']['palette']
        elif dataset_type == "Vaihingen":
            palette_rgb = _DATASET_METAINFO['ISPRSDataset']['palette']  # Vaihingen dataset
        else:
            # 默认使用iSAID的配色
            palette_rgb = _DATASET_METAINFO['iSAIDDataset']['palette']
        
        palette_rgb = np.array(palette_rgb, dtype=np.uint8)
        
        # 创建彩色分割图
        h, w = seg_pred.shape
        seg_colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 为每个类别分配颜色，确保使用官方配色方案中的颜色
        unique_labels = np.unique(seg_pred)
        for label in unique_labels:
            # 检查标签值是否在有效范围内
            if 0 <= label < len(palette_rgb):
                seg_colored[seg_pred == label] = palette_rgb[label]
            else:
                # 将超出范围的标签映射为黑色
                black_color = np.array([0, 0, 0], dtype=np.uint8)  # 黑色
                seg_colored[seg_pred == label] = black_color
        
        # 转换为PIL图像
        seg_image = Image.fromarray(seg_colored)
        
        # 确保输出尺寸与输入图像一致
        seg_image = seg_image.resize(pil_image.size)
        
        # 应用透明度混合
        # 将原始图像转换为numpy数组以便混合
        original_np = np.array(pil_image)
        # 确保尺寸一致
        if original_np.shape[:2] != seg_colored.shape[:2]:
            seg_colored = np.array(Image.fromarray(seg_colored).resize((original_np.shape[1], original_np.shape[0])))
            
        # Alpha混合: result = original * (1 - alpha) + segmented * alpha
        # transparency=0 -> 完全透明 (显示原图)
        # transparency=1 -> 完全不透明 (显示分割图)
        blended_np = (original_np * (1 - transparency) + seg_colored * transparency).astype(np.uint8)
        blended_image = Image.fromarray(blended_np)
        
        # 使用matplotlib进行可视化
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # 显示原始图像
        ax[0].imshow(pil_image)
        ax[0].axis('off')
        
        # 显示混合后的分割结果
        ax[1].imshow(blended_np)
        ax[1].axis('off')
        
        # 添加标题
        ax[0].set_title('Original Image')
        ax[1].set_title(f'Segmentation Result (Transparency: {transparency})')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存到临时文件
        temp_output = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(temp_output.name, bbox_inches='tight')
        plt.close()
        
        # 转换为PIL图像
        result_image = Image.open(temp_output.name)
        
        # 清理临时文件
        os.unlink(temp_output.name)
        os.unlink(temp_class_file)
        
        # 如果使用了扩展提示池，清理临时pickle文件
        if expanded_prompt_pool_path and os.path.exists(expanded_prompt_pool_path):
            os.unlink(expanded_prompt_pool_path)
        
        return result_image
    except Exception as e:
        # 清理临时文件
        if 'temp_class_file' in locals() and os.path.exists(temp_class_file):
            try:
                os.unlink(temp_class_file)
            except:
                pass
        if 'expanded_prompt_pool_path' in locals() and os.path.exists(expanded_prompt_pool_path):
            try:
                os.unlink(expanded_prompt_pool_path)
            except:
                pass
        raise e


def compare_segmentations(image, class_option, custom_prompts, device, transparency):
    """同时执行原始提示词和扩展提示词的分割，并显示对比"""
    if image is None:
        return None, "请上传一张图片"
    
    # 根据选项确定类别文件
    if class_option == "iSAID":
        base_class_file = "./configs/cls_iSAID.txt"
        exp_class_file = "./configs/cls_iSAID_exp.txt"
    elif class_option == "LoveDA":
        base_class_file = "./configs/cls_loveda.txt"
        exp_class_file = "./configs/cls_loveda_exp.txt"
    elif class_option == "Potsdam":
        base_class_file = "./configs/cls_potsdam.txt"
        exp_class_file = "./configs/cls_potsdam_exp.txt"
    elif class_option == "Vaihingen":
        base_class_file = "./configs/cls_vaihingen.txt"
        exp_class_file = "./configs/cls_vaihingen_exp.txt"
    else:
        return None, "请选择一个数据集"
    
    # 获取基础类别名称
    base_classes = get_class_names_from_file(base_class_file)
    
    # 如果用户输入了自定义提示词，则使用它们
    if custom_prompts.strip():
        base_query_words = [prompt.strip() for prompt in custom_prompts.split(',') if prompt.strip()]
        exp_query_words = base_query_words  # 自定义提示词时，两者相同
    else:
        base_query_words = base_classes
        exp_query_words = get_class_names_from_file(exp_class_file)
    
    try:
        # 执行原始提示词分割
        base_result = direct_segment_image(image, base_query_words, device, class_option, transparency, False)
        
        # 执行扩展提示词分割
        exp_result = direct_segment_image(image, exp_query_words, device, class_option, transparency, True)
        
        # 创建对比结果列表 [原图, 原始提示词结果, 扩展提示词结果]
        result_list = [image, base_result, exp_result]
        
        comparison_info = f"""对比结果:
        - 原始提示词: {len(base_query_words)} 个 ({', '.join(base_query_words[:5])}{'...' if len(base_query_words) > 5 else ''})
        - 扩展提示词: {len(exp_query_words)} 个 ({', '.join(exp_query_words[:5])}{'...' if len(exp_query_words) > 5 else ''})
        """
        
        return result_list, comparison_info
    
    except Exception as e:
        return [None], f"分割失败: {str(e)}"


def segment_image(image, class_option, custom_prompts, use_expanded_prompts, device, transparency):
    """执行图像分割"""
    if image is None:
        return None, "请上传一张图片"
    
    # 根据选项确定类别文件
    if class_option == "iSAID":
        base_class_file = "./configs/cls_iSAID.txt"
        exp_class_file = "./configs/cls_iSAID_exp.txt"
    elif class_option == "LoveDA":
        base_class_file = "./configs/cls_loveda.txt"
        exp_class_file = "./configs/cls_loveda_exp.txt"
    elif class_option == "Potsdam":
        base_class_file = "./configs/cls_potsdam.txt"
        exp_class_file = "./configs/cls_potsdam_exp.txt"
    elif class_option == "Vaihingen":
        base_class_file = "./configs/cls_vaihingen.txt"
        exp_class_file = "./configs/cls_vaihingen_exp.txt"
    else:
        return None, "请选择一个数据集"
    
    # 获取基础类别名称
    base_classes = get_class_names_from_file(base_class_file)
    
    # 如果用户输入了自定义提示词，则使用它们
    if custom_prompts.strip():
        query_words = [prompt.strip() for prompt in custom_prompts.split(',') if prompt.strip()]
    else:
        # 根据是否使用扩展提示词决定使用哪个列表
        if use_expanded_prompts:
            query_words = get_class_names_from_file(exp_class_file)
        else:
            query_words = base_classes
    
    # 如果使用扩展提示词，需要特别处理，确保类别与颜色映射正确
    if use_expanded_prompts and not custom_prompts.strip():
        # 从基础类名文件获取基础类别，用于正确映射颜色
        base_class_lines = []
        with open(base_class_file, 'r') as f:
            base_class_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # 创建类别到索引的映射
        class_to_idx = {}
        idx_counter = 0
        for line in base_class_lines:
            parts = [part.strip() for part in line.split(',') if part.strip()]
            for part in parts:
                class_to_idx[part] = idx_counter
            idx_counter += 1
    
    try:
        # 使用直接分割方法，传入use_expanded_prompts参数
        result_image = direct_segment_image(image, query_words, device, class_option, transparency, use_expanded_prompts)
        
        # 将单个图像转换为列表，以便与Gallery组件兼容
        result_list = [result_image]
        
        return result_list, f"成功完成分割，使用了 {len(query_words)} 个提示词: {', '.join(query_words[:5])}{'...' if len(query_words) > 5 else ''}"
    
    except Exception as e:
        return [None], f"分割失败: {str(e)}"
        

def analyze_image_with_qwen3vl(image_path, dataset_type):
    """使用Qwen3-VL分析图像"""
    try:
        # 初始化QwenAgent
        qwen_agent = QwenAgent(dataset_name=dataset_type)
        
        # 进行场景分析
        analysis_result = qwen_agent.analyze_scene(image_path, detail_level="high")
        
        return analysis_result
    except Exception as e:
        return f"Qwen3-VL分析失败: {str(e)}"


def generate_interactive_visualization(pil_image, query_words, device, dataset_type, transparency=0.5):
    """生成交互式可视化图表"""
    # 创建临时文件并执行分割
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for cls in query_words:
            f.write(cls + "\n")
        temp_class_file = f.name
    
    try:
        # 创建模型实例
        model = SegEarthOV3Segmentation(
            classname_path=temp_class_file,
            device=device,
            prob_thd=0.1,
            confidence_threshold=0.4,
            use_sem_seg=True,
            use_presence_score=True,
            use_transformer_decoder=True
        )
        
        # 获取分割结果
        seg_logits = model._inference_single_view(pil_image)
        aggregated_logits = aggregate_predictions_by_class(seg_logits, model.query_idx, model.num_cls)
        
        if model.num_cls != seg_logits.shape[0]:
            seg_pred = torch.from_numpy(aggregated_logits).argmax(0).cpu().numpy()
        else:
            seg_pred = torch.from_numpy(aggregated_logits).argmax(0).cpu().numpy()
        
        # 获取数据集配色方案
        if dataset_type == "iSAID":
            palette_rgb = _DATASET_METAINFO['iSAIDDataset']['palette']
        elif dataset_type == "LoveDA":
            palette_rgb = _DATASET_METAINFO['LoveDADataset']['palette']
        elif dataset_type == "Potsdam":
            palette_rgb = _DATASET_METAINFO['PotsdamDataset']['palette']
        elif dataset_type == "Vaihingen":
            palette_rgb = _DATASET_METAINFO['ISPRSDataset']['palette']
        else:
            palette_rgb = _DATASET_METAINFO['iSAIDDataset']['palette']
        
        palette_rgb = np.array(palette_rgb, dtype=np.uint8)
        
        # 计算各区域的面积占比
        unique, counts = np.unique(seg_pred, return_counts=True)
        total_pixels = seg_pred.size
        area_percentages = [(count / total_pixels) * 100 for count in counts]
        
        # 获取类别名称
        class_names = get_class_names_from_file(f"./configs/cls_{dataset_type.lower()}.txt")
        
        # 创建可视化图表
        fig = go.Figure()
        
        # 添加饼状图显示各类别面积占比
        fig.add_trace(go.Pie(
            labels=[class_names[i] if i < len(class_names) else f"Class {i}" for i in unique],
            values=area_percentages,
            name="Area Distribution"
        ))
        
        fig.update_layout(
            title=f"Segmentation Area Distribution - {dataset_type}",
            font=dict(size=14)
        )
        
        # 创建热力图显示空间分布
        heatmap_fig = px.imshow(
            seg_pred,
            title=f"Segmentation Heatmap - {dataset_type}",
            color_continuous_scale='viridis',
            labels={'x': 'Width', 'y': 'Height'}
        )
        
        # 返回两个图表
        return fig, heatmap_fig
    except Exception as e:
        print(f"生成交互式可视化失败: {str(e)}")
        return None, None
    finally:
        # 清理临时文件
        if os.path.exists(temp_class_file):
            os.remove(temp_class_file)


def smart_segment_and_analyze(image, class_option, custom_prompts, device, transparency, analyze_with_qwen):
    """智能分割与分析"""
    if image is None:
        return [None], "请上传一张图片"
    
    # 根据选项确定类别文件
    if class_option == "iSAID":
        class_file = "./configs/cls_iSAID.txt"
        exp_class_file = "./configs/cls_iSAID_exp.txt"
    elif class_option == "LoveDA":
        class_file = "./configs/cls_loveda.txt"
        exp_class_file = "./configs/cls_loveda_exp.txt"
    elif class_option == "Potsdam":
        class_file = "./configs/cls_potsdam.txt"
        exp_class_file = "./configs/cls_potsdam_exp.txt"
    elif class_option == "Vaihingen":
        class_file = "./configs/cls_vaihingen.txt"
        exp_class_file = "./configs/cls_vaihingen_exp.txt"
    else:
        return [None], "请选择一个数据集"
    
    # 获取类别名称
    if custom_prompts.strip():
        query_words = [prompt.strip() for prompt in custom_prompts.split(',') if prompt.strip()]
    else:
        query_words = get_class_names_from_file(class_file)
    
    # 执行分割
    try:
        result_image = direct_segment_image(image, query_words, device, class_option, transparency, False)
        
        # 生成交互式可视化
        pie_chart, heatmap = generate_interactive_visualization(image, query_words, device, class_option, transparency)
        
        # 使用Qwen3-VL进行分析
        analysis_result = ""
        if analyze_with_qwen:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                image.save(tmp_file.name)
                analysis_result = analyze_image_with_qwen3vl(tmp_file.name, class_option)
                os.remove(tmp_file.name)
        
        # 整理返回结果
        results = [image, result_image]
        if pie_chart:
            results.append(pie_chart)
        if heatmap:
            results.append(heatmap)
        
        message = f"成功完成分割，使用了 {len(query_words)} 个提示词"
        if analysis_result:
            message += f"\nQwen3-VL分析结果:\n{analysis_result}"
        
        return results, message
    except Exception as e:
        return [None], f"分割失败: {str(e)}"


def get_available_datasets():
    """返回可用的数据集列表"""
    datasets = ["iSAID", "LoveDA", "Potsdam", "Vaihingen"]
    return datasets


def get_available_devices():
    """返回可用的设备列表"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            devices.append(f"cuda:{i}")
    return devices


# 创建Gradio界面
with gr.Blocks(title="QwSAM3 智能分割可视化", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 QwSAM3 智能分割可视化平台")
    gr.Markdown("融合Qwen3-VL智能解译与SAM3精准分割的炫酷可视化界面")
    
    with gr.Tab("智能分割"):
        with gr.Row():
            with gr.Column(scale=1):
                # 数据集选择
                dataset_choice = gr.Dropdown(
                    choices=get_available_datasets(),
                    value="LoveDA",
                    label="选择数据集"
                )
                
                # 设备选择
                device_choice = gr.Dropdown(
                    choices=get_available_devices(),
                    value="cpu" if not torch.cuda.is_available() else "cuda:0",
                    label="选择设备"
                )
                
                # 透明度调节滑块
                transparency_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    label="分割结果透明度",
                    info="0为完全透明（类似GT风格），1为完全不透明"
                )
                
                # Qwen3-VL分析选项
                analyze_with_qwen = gr.Checkbox(label="启用Qwen3-VL智能解译", value=True)
                
                # 提示词选项
                custom_prompts_input = gr.Textbox(
                    label="自定义提示词 (用逗号分隔)",
                    placeholder="例如: building,house,structure 或留空使用预设提示词"
                )
                
                # 单张图片分割
                gr.Markdown("### 单张图片分割")
                image_input = gr.Image(type="pil", label="上传图片")
                smart_segment_btn = gr.Button("执行智能分割", variant="primary")
                
            with gr.Column(scale=2):
                # 输出区域 - 统一使用Gallery组件
                image_output = gr.Gallery(
                    label="分割结果与分析",
                    show_label=True,
                    columns=2,  # 两列显示
                    object_fit="contain",
                    height="auto"
                )
                status_output = gr.Textbox(label="状态信息", interactive=False, max_lines=10)
    
    with gr.Tab("对比分析"):
        with gr.Row():
            with gr.Column():
                # 数据集选择
                compare_dataset_choice = gr.Dropdown(
                    choices=get_available_datasets(),
                    value="LoveDA",
                    label="选择数据集"
                )
                
                # 设备选择
                compare_device_choice = gr.Dropdown(
                    choices=get_available_devices(),
                    value="cpu" if not torch.cuda.is_available() else "cuda:0",
                    label="选择设备"
                )
                
                # 透明度调节滑块
                compare_transparency_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    label="分割结果透明度",
                    info="0为完全透明（类似GT风格），1为完全不透明"
                )
                
                # 提示词选项
                compare_custom_prompts_input = gr.Textbox(
                    label="自定义提示词 (用逗号分隔)",
                    placeholder="例如: building,house,structure 或留空使用预设提示词"
                )
                
                # 单张图片分割
                gr.Markdown("### 单张图片对比")
                compare_image_input = gr.Image(type="pil", label="上传图片")
                compare_btn = gr.Button("执行对比分析", variant="primary")
                
            with gr.Column():
                # 对比输出区域
                compare_image_output = gr.Gallery(
                    label="对比结果",
                    show_label=True,
                    columns=3,  # 在对比模式下显示3列（原图、原始结果、扩展结果）
                    object_fit="contain",
                    height="auto"
                )
                compare_status_output = gr.Textbox(label="状态信息", interactive=False)
    
    # 智能分割事件绑定
    smart_segment_btn.click(
        fn=smart_segment_and_analyze,
        inputs=[image_input, dataset_choice, custom_prompts_input, device_choice, transparency_slider, analyze_with_qwen],
        outputs=[image_output, status_output]
    )
    
    # 对比分析事件绑定
    compare_btn.click(
        fn=compare_segmentations,
        inputs=[compare_image_input, compare_dataset_choice, compare_custom_prompts_input, compare_device_choice, compare_transparency_slider],
        outputs=[compare_image_output, compare_status_output]
    )

    gr.Markdown("## 📊 使用说明")
    gr.Markdown("""
    1. **智能分割标签页**：
       - 选择对应的数据集（iSAID, LoveDA, Potsdam, Vaihingen）
       - 选择计算设备（CPU或GPU）
       - 调节分割结果透明度
       - 上传图片并点击"执行智能分割"
       - 可启用Qwen3-VL智能解译获得场景分析
       - 查看分割结果、面积分布饼图和空间分布热力图
    
    2. **对比分析标签页**：
       - 比较原始提示词和扩展提示词的分割效果
       - 直观展示扩展提示词带来的改进
    
    3. **炫酷功能**：
       - Qwen3-VL智能场景解译
       - 交互式可视化图表
       - 面积分布统计
       - 空间分布热力图
    """)

# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)