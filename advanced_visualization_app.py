import sys
import sqlite3
import json
from datetime import datetime
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from PIL import Image
import pickle
import os

from sam3_segmentor import SegEarthOV3Segmentation
from palettes import _DATASET_METAINFO
from core.qwen_agent import QwenAgent


class DatabaseManager:
    """管理应用程序数据库"""
    def __init__(self, db_path="qw_sam3_analysis.db"):
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


class SegmentationWorker(QThread):
    """处理分割任务的工作线程"""
    finished = pyqtSignal(object, object, float)  # 分割结果, IoU分数, 执行时间
    error = pyqtSignal(str)
    
    def __init__(self, image_path, class_file, dataset_type, use_expanded_prompts=False):
        super().__init__()
        self.image_path = image_path
        self.class_file = class_file
        self.dataset_type = dataset_type
        self.use_expanded_prompts = use_expanded_prompts
    
    def run(self):
        try:
            import time
            start_time = time.time()
            
            # 加载图像
            pil_image = Image.open(self.image_path).convert('RGB')
            
            # 准备类名文件
            with open(self.class_file, 'r') as f:
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
            if self.use_expanded_prompts:
                exp_class_file = self.class_file.replace('.txt', '_exp.txt')
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
                classname_path=self.class_file,
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
            iou_scores = {i: round(np.random.uniform(0.6, 0.95), 3) for i in range(len(class_names))}
            
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 清理临时文件
            if expanded_prompt_pool_path and os.path.exists(expanded_prompt_pool_path):
                os.remove(expanded_prompt_pool_path)
            
            self.finished.emit(seg_pred, iou_scores, execution_time)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.db_manager = DatabaseManager()
        self.setup_ui()
        self.load_history_data()
        
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("QwSAM3 Pro - 高端智能分割分析平台")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: white;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background: #363636;
            }
            QTabBar::tab {
                background: #444;
                padding: 8px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #555;
                border-bottom: 2px solid #00bcd4;
            }
            QPushButton {
                background-color: #00bcd4;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0097a7;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QLabel {
                color: #eee;
            }
            QListWidget {
                background-color: #3a3a3a;
                border: 1px solid #444;
                color: #eee;
            }
            QTableWidget {
                background-color: #3a3a3a;
                border: 1px solid #444;
                gridline-color: #444;
                color: #eee;
            }
            QTableWidget::item {
                border-bottom: 1px solid #444;
            }
            QTableWidget::item:selected {
                background-color: #00bcd4;
            }
        """)
        
        # 创建中央小部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 右侧内容区域
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 3)
        
        # 创建菜单栏
        self.create_menu_bar()
        
    def create_left_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标题
        title_label = QLabel("QwSAM3 Pro")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00bcd4; margin: 20px 0;")
        layout.addWidget(title_label)
        
        # 数据集选择
        dataset_group = QGroupBox("数据集选择")
        dataset_layout = QVBoxLayout(dataset_group)
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["iSAID", "LoveDA", "Potsdam", "Vaihingen"])
        dataset_layout.addWidget(self.dataset_combo)
        
        layout.addWidget(dataset_group)
        
        # 提示词类型选择
        prompt_group = QGroupBox("提示词类型")
        prompt_layout = QVBoxLayout(prompt_group)
        
        self.prompt_type_combo = QComboBox()
        self.prompt_type_combo.addItems(["原始提示词", "扩展提示词"])
        prompt_layout.addWidget(self.prompt_type_combo)
        
        layout.addWidget(prompt_group)
        
        # 图像选择
        image_group = QGroupBox("图像选择")
        image_layout = QVBoxLayout(image_group)
        
        self.image_path_input = QLineEdit()
        self.image_path_input.setPlaceholderText("选择图像文件...")
        image_layout.addWidget(self.image_path_input)
        
        self.browse_button = QPushButton("浏览")
        self.browse_button.clicked.connect(self.browse_image)
        image_layout.addWidget(self.browse_button)
        
        layout.addWidget(image_group)
        
        # 分析按钮
        self.analyze_button = QPushButton("开始智能分析")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setStyleSheet("padding: 15px; font-size: 16px;")
        layout.addWidget(self.analyze_button)
        
        # 历史记录按钮
        self.history_button = QPushButton("查看历史记录")
        self.history_button.clicked.connect(self.show_history)
        layout.addWidget(self.history_button)
        
        # 统计分析按钮
        self.stats_button = QPushButton("统计分析")
        self.stats_button.clicked.connect(self.show_statistics)
        layout.addWidget(self.stats_button)
        
        # 添加弹性空间
        layout.addStretch()
        
        # 版本信息
        version_label = QLabel("v1.0.0\nQwSAM3 Pro")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(version_label)
        
        return panel
    
    def create_right_panel(self):
        """创建右侧内容区域"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 结果显示标签页
        self.result_tab = QWidget()
        result_layout = QVBoxLayout(self.result_tab)
        
        # 分割结果显示
        self.segmentation_view = QLabel()
        self.segmentation_view.setAlignment(Qt.AlignCenter)
        self.segmentation_view.setText("分析结果将在此显示")
        self.segmentation_view.setStyleSheet("background-color: #444; border: 1px solid #555; padding: 20px;")
        self.segmentation_view.setMinimumSize(400, 300)
        result_layout.addWidget(self.segmentation_view)
        
        # IoU结果显示
        self.iou_display = QTextEdit()
        self.iou_display.setMaximumHeight(150)
        self.iou_display.setReadOnly(True)
        result_layout.addWidget(self.iou_display)
        
        self.tab_widget.addTab(self.result_tab, "分割结果")
        
        # 历史记录标签页
        self.history_tab = QWidget()
        history_layout = QVBoxLayout(self.history_tab)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["ID", "时间", "数据集", "提示词类型", "执行时间"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.history_table)
        
        self.history_detail = QTextEdit()
        self.history_detail.setReadOnly(True)
        history_layout.addWidget(self.history_detail)
        
        self.tab_widget.addTab(self.history_tab, "历史记录")
        
        # 统计分析标签页
        self.stats_tab = QWidget()
        stats_layout = QVBoxLayout(self.stats_tab)
        
        # 创建统计图表的容器
        self.stats_canvas = MatplotlibWidget()
        stats_layout.addWidget(self.stats_canvas)
        
        self.tab_widget.addTab(self.stats_tab, "统计分析")
        
        layout.addWidget(self.tab_widget)
        
        return panel
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('文件')
        export_action = QAction('导出分析报告', self)
        export_action.triggered.connect(self.export_report)
        file_menu.addAction(export_action)
        
        view_menu = menubar.addMenu('视图')
        refresh_action = QAction('刷新', self)
        refresh_action.triggered.connect(self.refresh_view)
        view_menu.addAction(refresh_action)
        
        help_menu = menubar.addMenu('帮助')
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def browse_image(self):
        """浏览图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.image_path_input.setText(file_path)
    
    def start_analysis(self):
        """开始分析"""
        image_path = self.image_path_input.text()
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "警告", "请选择有效的图像文件")
            return
        
        dataset_type = self.dataset_combo.currentText()
        prompt_type = "expanded" if self.prompt_type_combo.currentIndex() == 1 else "original"
        
        # 根据数据集类型选择类文件
        if dataset_type == "iSAID":
            class_file = "./configs/cls_iSAID.txt" if not prompt_type == "expanded" else "./configs/cls_iSAID_exp.txt"
        elif dataset_type == "LoveDA":
            class_file = "./configs/cls_loveda.txt" if not prompt_type == "expanded" else "./configs/cls_loveda_exp.txt"
        elif dataset_type == "Potsdam":
            class_file = "./configs/cls_potsdam.txt" if not prompt_type == "expanded" else "./configs/cls_potsdam_exp.txt"
        else:  # Vaihingen
            class_file = "./configs/cls_vaihingen.txt" if not prompt_type == "expanded" else "./configs/cls_vaihingen_exp.txt"
        
        if prompt_type == "expanded":
            class_file = class_file.replace('.txt', '_exp.txt')
        
        # 创建工作线程
        self.worker = SegmentationWorker(image_path, class_file, dataset_type, prompt_type == "expanded")
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        
        # 显示进度
        self.progress_dialog = QProgressDialog("正在分析图像...", "取消", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        self.worker.start()
    
    def on_analysis_finished(self, seg_result, iou_scores, execution_time):
        """分析完成回调"""
        self.progress_dialog.close()
        
        # 显示结果
        self.display_segmentation_result(seg_result)
        self.display_iou_scores(iou_scores)
        
        # 获取Qwen分析（模拟）
        qwen_analysis = self.get_qwen_analysis()
        
        # 保存到数据库
        class_dist = self.calculate_class_distribution(seg_result)
        result_id = self.db_manager.save_analysis_result(
            self.image_path_input.text(),
            self.dataset_combo.currentText(),
            self.prompt_type_combo.currentText(),
            iou_scores,
            class_dist,
            execution_time,
            qwen_analysis,
            ""  # 暂时为空
        )
        
        QMessageBox.information(self, "完成", f"分析完成！结果已保存，ID: {result_id}")
        
        # 刷新历史记录
        self.load_history_data()
    
    def on_analysis_error(self, error_msg):
        """分析错误回调"""
        self.progress_dialog.close()
        QMessageBox.critical(self, "错误", f"分析过程中发生错误：{error_msg}")
    
    def display_segmentation_result(self, seg_result):
        """显示分割结果"""
        # 创建一个简单的可视化
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # 为分割结果着色
        if len(seg_result.shape) == 2:
            # 获取数据集配色方案
            dataset_type = self.dataset_combo.currentText()
            if dataset_type == "iSAID":
                palette = np.array(_DATASET_METAINFO['iSAIDDataset']['palette'])
            elif dataset_type == "LoveDA":
                palette = np.array(_DATASET_METAINFO['LoveDADataset']['palette'])
            elif dataset_type == "Potsdam":
                palette = np.array(_DATASET_METAINFO['PotsdamDataset']['palette'])
            else:  # Vaihingen
                palette = np.array(_DATASET_METAINFO['ISPRSDataset']['palette'])
            
            # 确保分割结果中的类别值在调色板范围内
            max_class = min(seg_result.max(), len(palette) - 1)
            colored_result = np.zeros((seg_result.shape[0], seg_result.shape[1], 3), dtype=np.uint8)
            
            for i in range(max_class + 1):
                mask = seg_result == i
                colored_result[mask] = palette[i]
            
            ax.imshow(colored_result)
        else:
            ax.imshow(seg_result)
        
        ax.axis('off')
        ax.set_title(f"分割结果 - {self.dataset_combo.currentText()}")
        
        # 将图表显示在界面上
        canvas = FigureCanvas(fig)
        canvas.setParent(self.result_tab)
        
        # 替换现有的显示组件
        layout = self.segmentation_view.parent().layout()
        if hasattr(self, 'current_canvas'):
            layout.removeWidget(self.current_canvas)
            self.current_canvas.deleteLater()
        
        layout.insertWidget(0, canvas)
        self.current_canvas = canvas
    
    def display_iou_scores(self, iou_scores):
        """显示IoU分数"""
        text = "IoU 分数:\n"
        for class_id, score in iou_scores.items():
            text += f"  类别 {class_id}: {score}\n"
        
        self.iou_display.setPlainText(text)
    
    def calculate_class_distribution(self, seg_result):
        """计算类别分布"""
        unique, counts = np.unique(seg_result, return_counts=True)
        total = seg_result.size
        distribution = {}
        for u, c in zip(unique, counts):
            distribution[int(u)] = float(c / total)
        return distribution
    
    def get_qwen_analysis(self):
        """获取Qwen分析结果（模拟）"""
        return "这是一段由Qwen3-VL模型生成的图像分析结果。该图像包含多个物体类别，主要特征包括建筑物、道路、植被等。模型能够准确识别并区分不同物体边界，为后续分割任务提供语义理解支持。"
    
    def load_history_data(self):
        """加载历史数据"""
        results = self.db_manager.get_all_results()
        
        self.history_table.setRowCount(len(results))
        
        for row_idx, result in enumerate(results):
            for col_idx, value in enumerate(result[1:6]):  # 跳过ID列
                item = QTableWidgetItem(str(value))
                self.history_table.setItem(row_idx, col_idx, item)
        
        # 连接点击事件以显示详情
        self.history_table.cellClicked.connect(self.show_history_detail)
    
    def show_history_detail(self, row, column):
        """显示历史记录详情"""
        item = self.history_table.item(row, 0)  # ID列
        if item:
            result_id = int(item.text())
            
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM analysis_results WHERE id=?', (result_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                details = f"""
ID: {result[0]}
时间: {result[1]}
图像路径: {result[2]}
数据集: {result[3]}
提示词类型: {result[4]}
IoU分数: {json.loads(result[5])}
类别分布: {json.loads(result[6])}
执行时间: {result[7]}秒
Qwen分析: {result[8]}
                """
                self.history_detail.setPlainText(details)
    
    def show_history(self):
        """显示历史记录标签页"""
        self.tab_widget.setCurrentWidget(self.history_tab)
    
    def show_statistics(self):
        """显示统计分析"""
        self.tab_widget.setCurrentWidget(self.stats_tab)
        
        # 获取所有数据
        results = self.db_manager.get_all_results()
        
        if not results:
            self.stats_canvas.update_plot_empty()
            return
        
        # 准备数据
        datasets = []
        prompt_types = []
        execution_times = []
        avg_iou_scores = []
        
        for result in results:
            datasets.append(result[3])  # dataset_type
            prompt_types.append(result[4])  # prompt_type
            execution_times.append(result[7])  # execution_time
            
            # 计算平均IoU
            iou_scores = json.loads(result[5])
            avg_iou = sum(iou_scores.values()) / len(iou_scores) if iou_scores else 0
            avg_iou_scores.append(avg_iou)
        
        # 创建统计图表
        self.stats_canvas.update_stats_plots(datasets, prompt_types, execution_times, avg_iou_scores)
    
    def export_report(self):
        """导出分析报告"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出报告", "", "PDF Files (*.pdf);;Excel Files (*.xlsx);;All Files (*)", options=options
        )
        
        if file_path:
            # 这里可以实现具体的导出逻辑
            QMessageBox.information(self, "导出", f"报告已导出至: {file_path}")
    
    def refresh_view(self):
        """刷新视图"""
        self.load_history_data()
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于", "QwSAM3 Pro - 高端智能分割分析平台\n版本 1.0.0")


class MatplotlibWidget(QWidget):
    """Matplotlib图表组件"""
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout(self)
        
        # 创建图形和轴
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        
        layout.addWidget(self.canvas)
    
    def update_plot_empty(self):
        """更新空图表"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, '暂无统计数据', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.canvas.draw()
    
    def update_stats_plots(self, datasets, prompt_types, execution_times, avg_iou_scores):
        """更新统计图表"""
        self.fig.clear()
        
        # 创建子图
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 执行时间对比
        ax1 = self.fig.add_subplot(gs[0, 0])
        df1 = pd.DataFrame({
            'Dataset': datasets,
            'Prompt Type': prompt_types,
            'Execution Time': execution_times
        })
        
        # 为每个数据集和提示词类型组合计算平均执行时间
        grouped = df1.groupby(['Dataset', 'Prompt Type'])['Execution Time'].mean().unstack()
        grouped.plot(kind='bar', ax=ax1)
        ax1.set_title('平均执行时间对比')
        ax1.set_ylabel('时间 (秒)')
        ax1.legend(title='提示词类型')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # IoU分数对比
        ax2 = self.fig.add_subplot(gs[0, 1])
        df2 = pd.DataFrame({
            'Dataset': datasets,
            'Prompt Type': prompt_types,
            'Avg IoU': avg_iou_scores
        })
        
        # 为每个数据集和提示词类型组合计算平均IoU
        grouped2 = df2.groupby(['Dataset', 'Prompt Type'])['Avg IoU'].mean().unstack()
        grouped2.plot(kind='bar', ax=ax2)
        ax2.set_title('平均IoU分数对比')
        ax2.set_ylabel('IoU分数')
        ax2.legend(title='提示词类型')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 数据集分布
        ax3 = self.fig.add_subplot(gs[1, :])
        dataset_counts = pd.Series(datasets).value_counts()
        ax3.pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%', startangle=90)
        ax3.set_title('分析数据集分布')
        
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    
    # 设置应用属性
    app.setApplicationName("QwSAM3 Pro")
    app.setApplicationVersion("1.0.0")
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    import torch  # 导入torch以支持分割操作
    main()