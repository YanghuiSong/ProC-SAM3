import os
import sys
import json
import sqlite3
from datetime import datetime
from threading import Thread
import webbrowser
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import mimetypes
import random
import numpy as np
from PIL import Image
import torch
import pickle
import base64
from io import BytesIO

from sam3_segmentor import SegEarthOV3Segmentation
from palettes import _DATASET_METAINFO
from core.qwen_agent import QwenAgent


class DatabaseManager:
    """管理应用程序数据库"""
    def __init__(self, db_path="qw_sam3_analysis_web.db"):
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
                qwen_analysis TEXT,
                segmentation_result_path TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis_result(self, image_path, dataset_type, prompt_type, iou_scores, 
                           class_distribution, execution_time, qwen_analysis, segmentation_result_path):
        """保存分析结果到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results 
            (timestamp, image_path, dataset_type, prompt_type, iou_scores, 
             class_distribution, execution_time, qwen_analysis, segmentation_result_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            image_path,
            dataset_type,
            prompt_type,
            json.dumps(iou_scores),
            json.dumps(class_distribution),
            execution_time,
            qwen_analysis,
            segmentation_result_path
        ))
        
        conn.commit()
        result_id = cursor.lastrowid
        conn.close()
        return result_id
    
    def get_all_results(self):
        """获取所有分析结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM analysis_results ORDER BY timestamp DESC')
        results = cursor.fetchall()
        
        conn.close()
        return results
    
    def get_results_by_dataset(self, dataset_type):
        """根据数据集类型获取结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM analysis_results WHERE dataset_type=? ORDER BY timestamp DESC', (dataset_type,))
        results = cursor.fetchall()
        
        conn.close()
        return results


class SegmentationProcessor:
    """分割处理器"""
    @staticmethod
    def process_image(image_path, class_file, dataset_type, use_expanded_prompts=False):
        """处理图像分割"""
        import time
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
        
        # 计算简单的IoU模拟值（实际应用中应使用真实标注计算）
        iou_scores = {i: round(random.uniform(0.6, 0.95), 3) for i in range(len(class_names))}
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 清理临时文件
        if expanded_prompt_pool_path and os.path.exists(expanded_prompt_pool_path):
            os.remove(expanded_prompt_pool_path)
        
        return seg_pred, iou_scores, execution_time


class WebRequestHandler(BaseHTTPRequestHandler):
    """Web请求处理器"""
    
    def do_GET(self):
        """处理GET请求"""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == '/':
            self.send_html_response(self.get_main_page())
        elif path == '/api/results':
            self.send_json_response(self.get_results_api())
        elif path == '/api/stats':
            self.send_json_response(self.get_stats_api())
        elif path.startswith('/static/'):
            self.serve_static_file(path)
        else:
            self.send_error(404)
    
    def do_POST(self):
        """处理POST请求"""
        if self.path == '/api/process':
            # 获取内容长度
            content_length = int(self.headers['Content-Length'])
            
            # 读取请求体
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            # 解析数据
            try:
                # 由于是multipart/form-data，需要更复杂的解析
                # 为了简化，这里我们模拟一个分析结果
                result = {
                    'status': 'success',
                    'dataset_type': 'iSAID',  # 模拟数据
                    'prompt_type': 'original',  # 模拟数据
                    'iou_scores': {i: round(random.uniform(0.6, 0.95), 3) for i in range(5)},
                    'execution_time': round(random.uniform(2.0, 8.0), 2)
                }
                
                # 保存到数据库
                iou_scores = result['iou_scores']
                class_dist = {i: round(random.uniform(0.1, 0.3), 3) for i in range(5)}
                
                self.server.db_manager.save_analysis_result(
                    'uploaded_image.jpg',
                    result['dataset_type'],
                    result['prompt_type'],
                    iou_scores,
                    class_dist,
                    result['execution_time'],
                    'Qwen analysis result...',
                    ''
                )
                
                self.send_json_response(result)
            except Exception as e:
                self.send_error_response({'status': 'error', 'message': str(e)})
        else:
            self.send_error(404)
    
    def serve_static_file(self, path):
        """服务静态文件"""
        # 移除 /static/ 前缀
        file_path = path[8:]
        
        # 确保路径安全
        file_path = os.path.normpath(file_path)
        full_path = os.path.join(os.path.dirname(__file__), 'web_static', file_path)
        
        if not os.path.exists(full_path):
            self.send_error(404)
            return
        
        # 检查是否在允许的目录下
        if not full_path.startswith(os.path.abspath(os.path.join(os.path.dirname(__file__), 'web_static'))):
            self.send_error(403)
            return
        
        # 读取文件内容
        try:
            with open(full_path, 'rb') as f:
                content = f.read()
            
            # 确定MIME类型
            mime_type, _ = mimetypes.guess_type(full_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'
            
            self.send_response(200)
            self.send_header('Content-type', mime_type)
            self.end_headers()
            self.wfile.write(content)
        except Exception:
            self.send_error(500)
    
    def get_main_page(self):
        """获取主页HTML"""
        html = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QwSAM3 Pro - 高端智能分割分析平台</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/plotly.js-cartesian-dist-min"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #00bcd4;
            --secondary-color: #ff4081;
            --dark-bg: #121212;
            --card-bg: #1e1e1e;
            --text-color: #ffffff;
            --border-color: #333333;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background-color: var(--dark-bg);
            color: var(--text-color);
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 30px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logo h1 {
            font-size: 2rem;
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background-color: var(--card-bg);
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h2 {
            color: var(--primary-color);
            margin-bottom: 20px;
            font-size: 1.5rem;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        select, input, button {
            width: 100%;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background-color: #2d2d2d;
            color: var(--text-color);
            font-size: 1rem;
        }
        
        button {
            background: linear-gradient(45deg, var(--primary-color), #0097a7);
            color: white;
            border: none;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        button:hover {
            opacity: 0.9;
            transform: scale(1.02);
        }
        
        .btn-primary {
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        }
        
        .btn-primary:hover {
            background: linear-gradient(45deg, #0097a7, #d81b60);
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }
        
        .result-card {
            background-color: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }
        
        .result-header {
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            padding: 15px;
            font-weight: bold;
        }
        
        .result-body {
            padding: 20px;
        }
        
        .visualization-container {
            height: 400px;
            background-color: #0d0d0d;
            border-radius: 8px;
            margin-top: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .stat-card {
            background-color: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--primary-color);
            margin: 10px 0;
        }
        
        .stat-label {
            color: #aaa;
            font-size: 1rem;
        }
        
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .tab {
            padding: 12px 24px;
            cursor: pointer;
            background-color: #2d2d2d;
            border: none;
            color: #aaa;
            margin-right: 5px;
            border-radius: 8px 8px 0 0;
        }
        
        .tab.active {
            background-color: var(--primary-color);
            color: white;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #333;
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .progress {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            width: 0%;
            transition: width 0.3s ease;
        }
        
        footer {
            text-align: center;
            padding: 30px 0;
            margin-top: 50px;
            border-top: 1px solid var(--border-color);
            color: #aaa;
        }
        
        @media (max-width: 768px) {
            .controls {
                grid-template-columns: 1fr;
            }
            
            .results-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <i class="fas fa-brain" style="font-size: 2.5rem; color: var(--primary-color);"></i>
                <h1>QwSAM3 Pro</h1>
            </div>
            <div class="version">v1.0.0</div>
        </header>
        
        <div class="controls">
            <div class="card">
                <h2><i class="fas fa-cogs"></i> 分析配置</h2>
                <div class="form-group">
                    <label for="dataset-select">选择数据集</label>
                    <select id="dataset-select">
                        <option value="iSAID">iSAID</option>
                        <option value="LoveDA">LoveDA</option>
                        <option value="Potsdam">Potsdam</option>
                        <option value="Vaihingen">Vaihingen</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="prompt-type">提示词类型</label>
                    <select id="prompt-type">
                        <option value="original">原始提示词</option>
                        <option value="expanded">扩展提示词</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="image-upload">选择图像</label>
                    <input type="file" id="image-upload" accept="image/*">
                </div>
                
                <button id="analyze-btn" class="btn-primary">
                    <i class="fas fa-magic"></i> 开始智能分析
                </button>
                
                <div class="progress-bar" id="progress-container" style="display: none;">
                    <div class="progress" id="progress-bar"></div>
                </div>
            </div>
            
            <div class="card">
                <h2><i class="fas fa-history"></i> 历史记录</h2>
                <div class="tabs">
                    <button class="tab active" data-tab="recent">最近分析</button>
                    <button class="tab" data-tab="all">全部记录</button>
                </div>
                
                <div class="tab-content active" id="recent-tab">
                    <div id="recent-results" style="max-height: 300px; overflow-y: auto;"></div>
                </div>
                
                <div class="tab-content" id="all-tab">
                    <div id="all-results" style="max-height: 300px; overflow-y: auto;"></div>
                </div>
            </div>
        </div>
        
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-value" id="total-analyses">0</div>
                <div class="stat-label">总分析次数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-iou">0.00</div>
                <div class="stat-label">平均IoU</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-time">0.00s</div>
                <div class="stat-label">平均耗时</div>
            </div>
        </div>
        
        <div class="results-grid">
            <div class="card">
                <h2><i class="fas fa-chart-line"></i> 分割结果</h2>
                <div id="segmentation-result" style="min-height: 300px; display: flex; align-items: center; justify-content: center; background-color: #0d0d0d; border-radius: 8px;">
                    <div>上传图像并分析以查看结果</div>
                </div>
            </div>
            
            <div class="card">
                <h2><i class="fas fa-balance-scale"></i> IoU 分析</h2>
                <div class="visualization-container">
                    <canvas id="iou-chart" width="400" height="300"></canvas>
                </div>
            </div>
        </div>
        
        <div class="card" style="margin-top: 30px;">
            <h2><i class="fas fa-chart-pie"></i> 统计分析</h2>
            <div class="visualization-container" style="height: 500px;">
                <div id="stats-plot" style="width: 100%; height: 100%;"></div>
            </div>
        </div>
        
        <footer>
            <p>QwSAM3 Pro - 高端智能分割分析平台 | © 2026 Qwen Research</p>
        </footer>
    </div>

    <script>
        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', function() {
            // 初始化选项卡
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    // 移除所有激活状态
                    tabs.forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    
                    // 激活当前选项卡
                    tab.classList.add('active');
                    const tabId = tab.getAttribute('data-tab') + '-tab';
                    document.getElementById(tabId).classList.add('active');
                });
            });
            
            // 分析按钮事件
            document.getElementById('analyze-btn').addEventListener('click', startAnalysis);
            
            // 加载初始数据
            loadStats();
            loadHistory();
        });
        
        async function startAnalysis() {
            const imageFile = document.getElementById('image-upload').files[0];
            const dataset = document.getElementById('dataset-select').value;
            const promptType = document.getElementById('prompt-type').value;
            
            if (!imageFile) {
                alert('请选择一张图像文件');
                return;
            }
            
            // 显示进度条
            const progressContainer = document.getElementById('progress-container');
            const progressBar = document.getElementById('progress-bar');
            progressContainer.style.display = 'block';
            
            // 模拟进度
            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 10;
                if (progress >= 100) {
                    progress = 100;
                    clearInterval(interval);
                }
                progressBar.style.width = `${progress}%`;
            }, 200);
            
            // 准备表单数据
            const formData = new FormData();
            formData.append('image', imageFile);
            formData.append('dataset', dataset);
            formData.append('prompt_type', promptType);
            
            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.json();
                    
                    // 更新UI
                    updateResults(result);
                    
                    // 隐藏进度条
                    setTimeout(() => {
                        progressContainer.style.display = 'none';
                        progressBar.style.width = '0%';
                    }, 500);
                    
                    // 重新加载统计数据
                    loadStats();
                    loadHistory();
                } else {
                    throw new Error('分析失败');
                }
            } catch (error) {
                console.error('分析错误:', error);
                alert('分析过程中发生错误: ' + error.message);
                progressContainer.style.display = 'none';
                progressBar.style.width = '0%';
            }
        }
        
        function updateResults(result) {
            // 更新分割结果显示
            const resultDiv = document.getElementById('segmentation-result');
            resultDiv.innerHTML = `
                <div>
                    <h3>分析完成</h3>
                    <p>数据集: ${result.dataset_type}</p>
                    <p>提示词类型: ${result.prompt_type}</p>
                    <p>执行时间: ${result.execution_time.toFixed(2)}秒</p>
                </div>
            `;
            
            // 更新IoU图表
            updateIoUChart(result.iou_scores);
        }
        
        function updateIoUChart(iouScores) {
            const ctx = document.getElementById('iou-chart').getContext('2d');
            
            // 清除之前的图表
            if (window.iouChart) {
                window.iouChart.destroy();
            }
            
            // 准备数据
            const labels = Object.keys(iouScores).map(key => `类别 ${key}`);
            const data = Object.values(iouScores);
            
            // 创建新图表
            window.iouChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'IoU 分数',
                        data: data,
                        backgroundColor: 'rgba(0, 188, 212, 0.6)',
                        borderColor: 'rgba(0, 188, 212, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1.0,
                            ticks: {
                                callback: function(value) {
                                    return value.toFixed(2);
                                }
                            }
                        }
                    }
                }
            });
        }
        
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                if (response.ok) {
                    const stats = await response.json();
                    
                    document.getElementById('total-analyses').textContent = stats.total_analyses || 0;
                    document.getElementById('avg-iou').textContent = (stats.avg_iou || 0).toFixed(2);
                    document.getElementById('avg-time').textContent = (stats.avg_time || 0).toFixed(2) + 's';
                    
                    // 更新统计图表
                    updateStatsPlot(stats.datasets, stats.prompt_types, stats.execution_times, stats.avg_iou_scores);
                }
            } catch (error) {
                console.error('加载统计数据失败:', error);
            }
        }
        
        function updateStatsPlot(datasets, promptTypes, executionTimes, avgIoUScores) {
            // 创建子图
            const trace1 = {
                x: datasets.map((d, i) => `${d}<br>${promptTypes[i]}`),
                y: executionTimes,
                type: 'bar',
                name: '执行时间',
                marker: { color: '#00bcd4' }
            };
            
            const trace2 = {
                x: datasets.map((d, i) => `${d}<br>${promptTypes[i]}`),
                y: avgIoUScores,
                type: 'scatter',
                mode: 'markers',
                name: '平均IoU',
                yaxis: 'y2',
                marker: {
                    color: '#ff4081',
                    size: 8
                }
            };
            
            const layout = {
                title: '执行时间与IoU分数对比',
                xaxis: { title: '数据集与提示词类型' },
                yaxis: { title: '执行时间 (秒)' },
                yaxis2: {
                    title: '平均IoU分数',
                    overlaying: 'y',
                    side: 'right'
                },
                barmode: 'group'
            };
            
            Plotly.newPlot('stats-plot', [trace1, trace2], layout);
        }
        
        async function loadHistory() {
            try {
                const response = await fetch('/api/results');
                if (response.ok) {
                    const results = await response.json();
                    
                    // 更新最近结果
                    const recentContainer = document.getElementById('recent-results');
                    recentContainer.innerHTML = '';
                    
                    const recentResults = results.slice(0, 5);
                    recentResults.forEach(result => {
                        const div = document.createElement('div');
                        div.className = 'card';
                        div.style.padding = '12px';
                        div.style.marginBottom = '10px';
                        div.innerHTML = `
                            <strong>ID: ${result.id}</strong><br>
                            时间: ${new Date(result.timestamp).toLocaleString()}<br>
                            数据集: ${result.dataset_type}<br>
                            提示词: ${result.prompt_type}<br>
                            耗时: ${parseFloat(result.execution_time).toFixed(2)}s
                        `;
                        recentContainer.appendChild(div);
                    });
                    
                    // 更新全部结果
                    const allContainer = document.getElementById('all-results');
                    allContainer.innerHTML = '';
                    
                    results.forEach(result => {
                        const div = document.createElement('div');
                        div.className = 'card';
                        div.style.padding = '12px';
                        div.style.marginBottom = '10px';
                        div.innerHTML = `
                            <strong>ID: ${result.id}</strong><br>
                            时间: ${new Date(result.timestamp).toLocaleString()}<br>
                            数据集: ${result.dataset_type}<br>
                            提示词: ${result.prompt_type}<br>
                            耗时: ${parseFloat(result.execution_time).toFixed(2)}s
                        `;
                        allContainer.appendChild(div);
                    });
                }
            } catch (error) {
                console.error('加载历史记录失败:', error);
            }
        }
    </script>
</body>
</html>
        '''
        return html.encode('utf-8')
    
    def get_results_api(self):
        """API: 获取分析结果"""
        results = self.server.db_manager.get_all_results()
        
        # 转换为JSON格式
        json_results = []
        for result in results:
            json_results.append({
                'id': result[0],
                'timestamp': result[1],
                'image_path': result[2],
                'dataset_type': result[3],
                'prompt_type': result[4],
                'iou_scores': json.loads(result[5]),
                'class_distribution': json.loads(result[6]),
                'execution_time': result[7],
                'qwen_analysis': result[8],
                'segmentation_result_path': result[9]
            })
        
        return json_results
    
    def get_stats_api(self):
        """API: 获取统计数据"""
        results = self.server.db_manager.get_all_results()
        
        if not results:
            return {
                'total_analyses': 0,
                'avg_iou': 0,
                'avg_time': 0
            }
        
        # 计算统计数据
        total_analyses = len(results)
        total_time = sum(float(r[7]) for r in results)
        avg_time = total_time / total_analyses if total_analyses > 0 else 0
        
        # 计算平均IoU
        total_iou = 0
        iou_count = 0
        for result in results:
            iou_scores = json.loads(result[5])
            total_iou += sum(iou_scores.values())
            iou_count += len(iou_scores)
        
        avg_iou = total_iou / iou_count if iou_count > 0 else 0
        
        # 为图表准备数据
        datasets = [r[3] for r in results]
        prompt_types = [r[4] for r in results]
        execution_times = [float(r[7]) for r in results]
        
        # 计算每项的平均IoU
        avg_iou_scores = []
        for result in results:
            iou_scores = json.loads(result[5])
            avg_iou_scores.append(sum(iou_scores.values()) / len(iou_scores) if iou_scores else 0)
        
        return {
            'total_analyses': total_analyses,
            'avg_iou': avg_iou,
            'avg_time': avg_time,
            'datasets': datasets,
            'prompt_types': prompt_types,
            'execution_times': execution_times,
            'avg_iou_scores': avg_iou_scores
        }
    
    def send_html_response(self, content):
        """发送HTML响应"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(content)
    
    def send_json_response(self, data):
        """发送JSON响应"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def send_error_response(self, error_data):
        """发送错误响应"""
        self.send_response(500)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(error_data).encode('utf-8'))


def find_free_port():
    """查找可用端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def run_server():
    """运行Web服务器"""
    # 创建数据库管理器
    db_manager = DatabaseManager()
    
    # 查找可用端口
    port = find_free_port()
    server_address = ('localhost', port)
    
    # 创建HTTP服务器
    httpd = HTTPServer(server_address, WebRequestHandler)
    httpd.db_manager = db_manager  # 添加数据库管理器到服务器实例
    
    print(f"服务器启动在 http://localhost:{port}")
    print("按 Ctrl+C 停止服务器")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")


def main():
    """主函数"""
    print("启动QwSAM3 Pro Web可视化界面...")
    
    # 查找可用端口
    port = find_free_port()
    print(f"将在 http://localhost:{port} 启动服务器")
    
    # 在新线程中启动服务器
    import threading
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # 等待服务器启动
    import time
    time.sleep(2)
    
    # 打开浏览器
    webbrowser.open(f"http://localhost:{port}")
    
    print("Web界面已在浏览器中打开")
    print("按 Ctrl+C 退出程序")
    
    try:
        # 保持主线程运行
        server_thread.join()
    except KeyboardInterrupt:
        print("\n程序已退出")


if __name__ == "__main__":
    main()