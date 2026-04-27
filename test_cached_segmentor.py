"""
测试新的缓存分割器功能
"""
import os
import torch
from PIL import Image
import time
from sam3_segmentor import SegEarthOV3Segmentation
from sam3_segmentor_cached import CachedSAM3OpenSegmentor
from core.prompt_bank import PromptBank


def test_vaihingen_cached_segmentor():
    """
    测试Vaihingen数据集上的缓存分割器性能
    """
    print("Testing CachedSAM3OpenSegmentor for Vaihingen dataset...")
    
    # 设置设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 准备类别文件路径
    classname_path = "./configs/cls_vaihingen.txt"
    
    # 检查类别文件是否存在
    if not os.path.exists(classname_path):
        print(f"Classname file not found: {classname_path}")
        return
    
    # 创建两个分割器实例进行对比测试
    print("\n1. Creating standard SegEarthOV3Segmentation...")
    standard_segmentor = SegEarthOV3Segmentation(
        classname_path=classname_path,
        device=device,
        prob_thd=0.0,
        confidence_threshold=0.5,
        use_sem_seg=True,
        use_presence_score=True,
        use_transformer_decoder=True
    )
    
    print("\n2. Creating CachedSAM3OpenSegmentor...")
    cached_segmentor = CachedSAM3OpenSegmentor(
        classname_path=classname_path,
        device=device,
        prob_thd=0.0,
        confidence_threshold=0.5,
        use_sem_seg=True,
        use_presence_score=True,
        use_transformer_decoder=True,
        cache_text_embeddings=True
    )
    
    # 创建一个模拟的PIL图像用于测试
    print("\n3. Creating test image...")
    test_image = Image.new('RGB', (512, 512), color='green')
    
    # 测试标准分割器的推理时间
    print("\n4. Testing standard segmentor inference time...")
    start_time = time.time()
    for i in range(3):  # 运行3次以观察性能差异
        result_standard = standard_segmentor._inference_single_view(test_image)
        print(f"Standard segmentor - Run {i+1} completed")
    standard_time = time.time() - start_time
    print(f"Standard segmentor total time: {standard_time:.2f}s")
    
    # 测试缓存分割器的推理时间
    print("\n5. Testing cached segmentor inference time...")
    start_time = time.time()
    for i in range(3):  # 运行3次以观察性能差异
        result_cached = cached_segmentor._inference_single_view(test_image)
        print(f"Cached segmentor - Run {i+1} completed")
    cached_time = time.time() - start_time
    print(f"Cached segmentor total time: {cached_time:.2f}s")
    
    # 比较结果
    print(f"\n6. Performance Comparison:")
    print(f"Standard segmentor: {standard_time:.2f}s for 3 runs")
    print(f"Cached segmentor: {cached_time:.2f}s for 3 runs")
    print(f"Difference: {standard_time - cached_time:.2f}s")
    print(f"Speedup ratio: {standard_time / cached_time:.2f}x" if cached_time > 0 else "N/A")
    
    # 检查结果是否相似
    print(f"\n7. Result similarity check:")
    print(f"Standard result shape: {result_standard.shape}")
    print(f"Cached result shape: {result_cached.shape}")
    
    # 简单的形状检查
    if result_standard.shape == result_cached.shape:
        print("✓ Shapes match")
    else:
        print("✗ Shapes do not match")
    
    # 验证缓存机制
    print(f"\n8. Text embedding cache verification:")
    cache_size = len(PromptBank._text_embedding_cache)
    print(f"Text embedding cache size: {cache_size}")
    
    if cache_size > 0:
        print("✓ Text embeddings are being cached")
    else:
        print("? Text embeddings cache is empty - may need to check implementation")
    
    print("\nTest completed!")


if __name__ == "__main__":
    test_vaihingen_cached_segmentor()
