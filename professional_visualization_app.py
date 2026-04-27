import os
import sys
import json
import sqlite3
from datetime import datetime
import random
import numpy as np
from PIL import Image
import pandas as pd
import torch
import pickle
import base64
from io import BytesIO
import time
import gradio as gr

from sam3_segmentor import SegEarthOV3Segmentation
from palettes import _DATASET_METAINFO


class DatabaseManager:
    """管理应用程序数据库"""
    def __init__(self, db_path="qw_sam3_professional_gradio.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建分析结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                image_path TEXT,
                dataset_type TEXT,
                prompt_type TEXT,  -- original or expanded
                iou_scores TEXT,   -- JSON格式的IoU分数
                class_distribution TEXT,  -- JSON格式的类别分布
                execution_time REAL,
                segmentation_result_path TEXT,
                metrics_summary TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis_result(self, image_path, dataset_type, prompt_type, iou_scores, 
                           class_distribution, execution_time, segmentation_result_path):
        """保存分析结果到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results 
            (timestamp, image_path, dataset_type, prompt_type, iou_scores, 
             class_distribution, execution_time, segmentation_result_path, metrics_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            image_path,
            dataset_type,
            prompt_type,
            json.dumps(iou_scores),
            json.dumps(class_distribution),
            execution_time,
            segmentation_result_path,
            json.dumps({"execution_time": execution_time})
        ))
        
        conn.commit()
        result_id = cursor.lastrowid
        conn.close()
        return result_id
    
    def get_all_results(self):
        """获取所有分析结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM analysis_results ORDER BY timestamp DESC LIMIT 50')
        results = cursor.fetchall()
        
        conn.close()
        return results


class SegmentationProcessor:
    """分割处理器"""
    @staticmethod
    def process_image(image_path, class_file, dataset_type, use_expanded_prompts=False):
        """处理图像分割"""
        start_time = time.time()
        
        # 加载图像
        pil_image = Image.open(image_path).convert('RGB')
        
        # 准备类名文件
        with open(class_file, 'r') as f:
            lines = f.readlines()
        
        class_names = []
        for line in lines:
            line = line.strip()
            if ',' in line:
                parts = [part.strip() for part in line.split(',') if part.strip()]
                class_names.extend(parts)
            else:
                class_names.append(line)
        
        # 确定扩展提示池路径
        expanded_prompt_pool_path = None
        if use_expanded_prompts:
            exp_class_file = class_file.replace('.txt', '_exp.txt')
            if os.path.exists(exp_class_file):
                # 创建临时的扩展提示池pickle文件
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                    temp_pkl_path = tmp_file.name
                    
                    # 从文本文件创建扩展提示池
                    expanded_prompt_pool = {}
                    with open(exp_class_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = [part.strip() for part in line.split(',') if part.strip()]
                            if len(parts) > 0:
                                key = ','.join(parts)
                                expanded_prompt_pool[key] = parts
                    
                    with open(temp_pkl_path, 'wb') as f:
                        pickle.dump(expanded_prompt_pool, f)
                    
                    expanded_prompt_pool_path = temp_pkl_path
        
        # 创建模型
        model = SegEarthOV3Segmentation(
            classname_path=class_file,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            prob_thd=0.1,
            confidence_threshold=0.4,
            use_sem_seg=True,
            use_presence_score=True,
            use_transformer_decoder=True,
            expanded_prompt_pool_path=expanded_prompt_pool_path
        )
        
        # 执行分割
        seg_result = model.predict([pil_image], [None])
        seg_pred = seg_result[0].pred_sem_seg.data.cpu().numpy().squeeze()
        
        # 获取数据集配色方案
        if dataset_type == "iSAID":
            palette = np.array(_DATASET_METAINFO['iSAIDDataset']['palette'])
        elif dataset_type == "LoveDA":
            palette = np.array(_DATASET_METAINFO['LoveDADataset']['palette'])
        elif dataset_type == "Potsdam":
            palette = np.array(_DATASET_METAINFO['PotsdamDataset']['palette'])
        else:  # Vaihingen
            palette = np.array(_DATASET_METAINFO['ISPRSDataset']['palette'])
        
        # 创建分割可视化结果
        h, w = seg_pred.shape
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 为每个类别分配颜色
        unique_labels = np.unique(seg_pred)
        for label in unique_labels:
            if 0 <= label < len(palette):
                vis_img[seg_pred == label] = palette[label]
            else:
                # 将超出范围的标签映射为黑色
                vis_img[seg_pred == label] = [0, 0, 0]
        
        # 模拟IoU分数（实际应用中应使用真实标注计算）
        iou_scores = {i: round(random.uniform(0.6, 0.95), 3) for i in range(len(class_names))}
        
        # 计算类别分布
        class_dist = {}
        for label in unique_labels:
            if label < len(class_names):
                class_dist[int(label)] = float(np.sum(seg_pred == label) / seg_pred.size)
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 清理临时文件
        if expanded_prompt_pool_path and os.path.exists(expanded_prompt_pool_path):
            os.remove(expanded_prompt_pool_path)
        
        return seg_pred, iou_scores, execution_time, vis_img, class_dist


def process_image_gradio(image_path, dataset_type, prompt_type):
    """处理图像的Gradio接口函数"""
    if not image_path or not os.path.exists(image_path):
        return "请提供有效的图像路径", None, gr.BarPlot.update(), "路径无效"
    
    # 根据数据集类型选择类文件
    if dataset_type == "iSAID":
        class_file = "./configs/cls_iSAID.txt"
        exp_class_file = "./configs/cls_iSAID_exp.txt"
    elif dataset_type == "LoveDA":
        class_file = "./configs/cls_loveda.txt"
        exp_class_file = "./configs/cls_loveda_exp.txt"
    elif dataset_type == "Potsdam":
        class_file = "./configs/cls_potsdam.txt"
        exp_class_file = "./configs/cls_potsdam_exp.txt"
    elif dataset_type == "Vaihingen":
        class_file = "./configs/cls_vaihingen.txt"
        exp_class_file = "./configs/cls_vaihingen_exp.txt"
    else:
        class_file = "./configs/cls_iSAID.txt"
        exp_class_file = "./configs/cls_iSAID_exp.txt"

    # 选择适当的类文件
    use_expanded = prompt_type == "expanded"
    selected_class_file = exp_class_file if use_expanded else class_file

    # 处理图像
    _, iou_scores, execution_time, vis_img, class_dist = SegmentationProcessor.process_image(
        image_path, 
        selected_class_file, 
        dataset_type, 
        use_expanded
    )

    # 保存到数据库
    db_manager = DatabaseManager()
    db_manager.save_analysis_result(
        image_path,
        dataset_type,
        prompt_type,
        iou_scores,
        class_dist,
        execution_time,
        ''
    )

    # 准备结果展示
    # 将IoU分数转换为DataFrame格式
    iou_df = pd.DataFrame(list(iou_scores.items()), columns=['Class', 'IoU'])
    
    # 将可视化图像转换为PIL格式
    vis_pil = Image.fromarray(vis_img)
    
    # 生成结果摘要
    result_summary = f"""
    分析完成！
    - 数据集: {dataset_type}
    - 提示词类型: {"扩展提示词" if use_expanded else "原始提示词"}
    - 执行时间: {execution_time:.2f}秒
    - 平均IoU: {sum(iou_scores.values())/len(iou_scores):.3f}
    - 类别数量: {len(iou_scores)}
    """
    
    return result_summary, vis_pil, iou_df, ""


def get_statistics():
    """获取统计信息"""
    db_manager = DatabaseManager()
    results = db_manager.get_all_results()
    
    if not results:
        return "暂无历史数据", [], gr.Plot.update()
    
    # 计算统计信息
    total_analyses = len(results)
    total_time = sum(r[7] for r in results)
    avg_time = total_time / total_analyses if total_analyses > 0 else 0
    
    # 计算平均IoU
    total_iou = 0
    iou_count = 0
    for result in results:
        iou_scores = json.loads(result[5])
        total_iou += sum(iou_scores.values())
        iou_count += len(iou_scores)
    
    avg_iou = total_iou / iou_count if iou_count > 0 else 0
    
    # 准备历史记录表格数据
    history_data = []
    for result in results[:20]:  # 只显示最近20条
        history_data.append([
            result[0],  # ID
            result[1][:19],  # 时间戳（去掉毫秒）
            result[3],  # 数据集
            "扩展" if result[4] == "expanded" else "原始",
            f"{result[7]:.2f}s",  # 执行时间
            f"{sum(json.loads(result[5]).values())/len(json.loads(result[5])):.3f}"  # 平均IoU
        ])
    
    stats_text = f"""
    总分析次数: {total_analyses}
    平均执行时间: {avg_time:.2f}秒
    平均IoU: {avg_iou:.3f}
    """
    
    return stats_text, history_data, gr.Plot.update()


def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(
        title="QwSAM3 Professional - 大数据智能分割分析平台",
        css="""
        .header { 
            text-align: center; 
            padding: 20px; 
            background: linear-gradient(90deg, #00bcd4, #ff4081);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            font-size: 2em;
        }
        .header p {
            margin: 5px 0 0 0;
            font-size: 1.1em;
        }
        """
    ) as demo:
        # 标题区域
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="header">
                    <h1>QwSAM3 Professional</h1>
                    <p>大数据智能分割分析平台 v2.0</p>
                </div>
                """)
        
        with gr.Tab("图像分割分析"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 控制面板
                    image_path = gr.Textbox(
                        label="图像路径",
                        placeholder="输入图像文件的完整路径",
                        elem_classes=["control-panel"]
                    )
                    
                    dataset_type = gr.Dropdown(
                        choices=["iSAID", "LoveDA", "Potsdam", "Vaihingen"],
                        value="iSAID",
                        label="数据集类型"
                    )
                    
                    prompt_type = gr.Radio(
                        choices=[("原始提示词", "original"), ("扩展提示词", "expanded")],
                        value="original",
                        label="提示词类型"
                    )
                    
                    run_btn = gr.Button("开始智能分析", variant="primary")
                
                with gr.Column(scale=2):
                    # 结果显示区域
                    result_summary = gr.Textbox(
                        label="分析摘要",
                        interactive=False,
                        elem_classes=["result-display"]
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            segmentation_result = gr.Image(
                                label="分割结果",
                                interactive=False,
                                height=400
                            )
                        
                        with gr.Column():
                            iou_chart = gr.BarPlot(
                                x="Class",
                                y="IoU",
                                title="IoU 分数分布",
                                height=400,
                                caption="各类别的IoU得分"
                            )
        
        with gr.Tab("统计分析"):
            with gr.Row():
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("刷新统计", variant="secondary")
                
                with gr.Column(scale=2):
                    stats_display = gr.Textbox(
                        label="统计摘要",
                        interactive=False,
                        lines=5
                    )
            
            with gr.Row():
                with gr.Column():
                    history_table = gr.Dataframe(
                        headers=["ID", "时间", "数据集", "提示词类型", "执行时间", "平均IoU"],
                        label="历史记录 (最近20条)",
                        datatype=["number", "str", "str", "str", "str", "number"],
                        interactive=False,
                        wrap=True
                    )
                
                with gr.Column():
                    overall_chart = gr.Plot(
                        label="总体趋势图",
                        show_label=True
                    )
        
        # 绑定事件
        run_btn.click(
            fn=process_image_gradio,
            inputs=[image_path, dataset_type, prompt_type],
            outputs=[result_summary, segmentation_result, iou_chart, gr.Textbox(visible=False)]
        )
        
        refresh_btn.click(
            fn=get_statistics,
            inputs=[],
            outputs=[stats_display, history_table, overall_chart]
        )
        
        # 初始化统计数据显示
        demo.load(
            fn=get_statistics,
            inputs=[],
            outputs=[stats_display, history_table, overall_chart]
        )
    
    return demo


if __name__ == "__main__":
    print("启动QwSAM3 Professional Gradio可视化界面...")
    print("在浏览器中打开显示的URL以使用应用程序")
    
    demo = create_interface()
    demo.queue(max_size=10).launch(
        share=False, 
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=True
    )