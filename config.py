# config.py
import os
import torch

class Config:
    # 模型路径
    QWEN_MODEL_PATH = "/data/public/Qwen3-VL-8B-Instruct"
    SAM3_CHECKPOINT = "/data/public/sam3/sam3.pt"
    
    # 数据集配置
    CLASS_FILE = "configs/cls_vaihingen.txt"  # 默认类别文件
    DATA_ROOT = "./data/"  # 默认数据根目录
    DEFAULT_CONF_THRESHOLD = 0.5  # 默认置信度阈值
    
    # 设备配置 - 支持通过环境变量指定
    @staticmethod
    def get_qwen_device():
        """获取千问模型设备字符串"""
        return Config.get_device()
    
    @staticmethod
    def get_sam3_device():
        """获取SAM3设备字符串"""
        return Config.get_device()
    
    # 指定使用的设备ID（默认使用第一张显卡，支持通过环境变量覆盖）
    DEVICE_ID = int(os.getenv("CUDA_DEVICE_ID", "0"))
    
    @staticmethod
    def get_device():
        """使用指定的单张显卡，返回设备字符串"""
        if torch.cuda.is_available():
            # 检查指定显卡是否存在
            if Config.DEVICE_ID < torch.cuda.device_count():
                device = f"cuda:{Config.DEVICE_ID}"
                
                # 检查显存情况
                try:
                    total_memory = torch.cuda.get_device_properties(Config.DEVICE_ID).total_memory / (1024**3)
                    allocated_memory = torch.cuda.memory_allocated(Config.DEVICE_ID) / (1024**3)
                    cached_memory = torch.cuda.memory_reserved(Config.DEVICE_ID) / (1024**3)
                    free_memory = total_memory - allocated_memory
                    
                    print(f"使用显卡: {device}")
                    print(f"显存统计 - 总量: {total_memory:.1f}GB, "
                          f"已分配: {allocated_memory:.1f}GB, "
                          f"缓存: {cached_memory:.1f}GB, "
                          f"剩余: {free_memory:.1f}GB")
                    
                    # 如果显存紧张，给出警告
                    if free_memory < 4:  # 少于4GB空闲显存
                        print(f"警告: 显存紧张 ({free_memory:.1f}GB)，可能需要优化模型加载")
                        
                except Exception as e:
                    print(f"显存检测失败: {e}")
                    
                return device
            else:
                print(f"指定的显卡 cuda:{Config.DEVICE_ID} 不可用")
                if torch.cuda.device_count() > 0:
                    print(f"可用显卡: {torch.cuda.device_count()}张，将使用 cuda:0")
                    return "cuda:0"
                else:
                    print("没有可用显卡，使用CPU")
                    return "cpu"
        else:
            print("CUDA不可用，使用CPU")
            return "cpu"
    
    @staticmethod
    def get_qwen_dtype():
        """根据设备选择合适的数据类型"""
        device = Config.get_device()
        if device == "cpu":
            return "float32"
        else:
            # 检查是否支持bfloat16
            if torch.cuda.is_bf16_supported():
                return "bfloat16"
            else:
                return "float16"
    
    # 系统配置
    TEMP_DIR = "./temp_results"
    
# 指令模板 - 英文增强版
    INSTRUCTION_TEMPLATE = """You are a professional visual segmentation assistant. Analyze the image and generate executable segmentation instruction sequences for SAM3.

CRITICAL GUIDELINES:
1. Generate instructions in ENGLISH ONLY
2. Use simple, concrete object names (e.g., "car", "building", "person", "tree")
3. Prioritize identifying ALL major objects in the image
4. Consider spatial relationships and occlusions
5. Use most effective prompt types

SAM3 supports these prompt types:
1. TEXT prompts: Single words or simple phrases (e.g., "car", "building", "person")
2. BOX prompts: Normalized coordinates [x_center, y_center, width, height] (0-1 range)
3. POINT prompts: Normalized coordinates [x, y] (0-1 range)

IMPORTANT: Generate JSON format instruction sequence EXACTLY as shown below:
{{
    "strategy": "Brief description of your segmentation strategy",
    "instructions": [
        {{
            "type": "text|box|point",
            "content": "Prompt content",
            "label": 1  // Only for box/point: 1=positive, 0=negative
        }}
    ]
}}

Current image description: {image_description}

User segmentation request: {user_request}

Analyze the image content, design optimal segmentation strategy, generate executable instruction sequence:"""