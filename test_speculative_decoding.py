#!/usr/bin/env python3
"""
测试简化版Speculative Decoding
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.edge_model import EdgeModel
from src.models.cloud_model import CloudModel
from src.speculative_decoding import SimpleSpeculativeDecoding

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data():
    """加载测试数据"""
    manifest_path = "data/processed/secap/manifest.json"
    
    if not os.path.exists(manifest_path):
        print(f"❌ 数据文件不存在: {manifest_path}")
        return None
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    if not manifest:
        print("❌ 数据文件为空")
        return None
    
    # 取第一个样本
    sample = manifest[0]
    
    # 加载音频数据
    from src.data.audio_processor import AudioProcessor
    audio_processor = AudioProcessor()
    
    try:
        audio_waveform = audio_processor.load_audio(sample['audio_path'])
        print(f"✅ 成功加载音频: {sample['audio_path']}")
    except Exception as e:
        print(f"❌ 音频加载失败: {e}")
        return None
    
    return {
        'file_id': sample['file_id'],
        'audio_waveform': audio_waveform,
        'reference_caption': sample.get('chinese_caption', sample.get('caption', '')),
        'prompt': "基于这个音频，用中文描述说话人的情感状态。"
    }


def compare_with_baselines(edge_model, cloud_model):
    """与baseline对比"""
    print("\n🔄 与baseline对比")
    print("=" * 60)
    
    # 加载测试数据
    test_data = load_test_data()
    if not test_data:
        return False
    
    try:
        # 使用传入的已初始化模型
        print("使用已初始化的模型进行对比...")
        
        spec_decoding = SimpleSpeculativeDecoding(
            edge_model, 
            cloud_model,
            entropy_threshold=0.6,  # 提高阈值，减少Cloud调用
            prob_threshold=0.25,  # 降低阈值，提高接受率
            k=3  # 减少k值，减少每次draft的token数量
        )
        
        # Edge baseline
        print("--- Edge Baseline ---")
        start_time = time.time()
        edge_text, edge_metrics = edge_model.generate_draft(
            test_data['audio_waveform'],
            test_data['prompt'],
            max_new_tokens=32,
            use_streaming=True
        )
        edge_time = time.time() - start_time
        
        print(f"Edge生成文本: {edge_text}")
        print(f"Edge时间: {edge_time:.4f}s")
        print(f"Edge TTFT: {edge_metrics.get('ttft', 'N/A'):.4f}s")
        print(f"Edge OTPS: {edge_metrics.get('otps', 'N/A'):.2f} tokens/s")
        
        # Cloud baseline
        print("\n--- Cloud Baseline ---")
        start_time = time.time()
        cloud_text, cloud_metrics = cloud_model.generate_independently(
            test_data['audio_waveform'],
            test_data['prompt'],
            max_new_tokens=32
        )
        cloud_time = time.time() - start_time
        
        print(f"Cloud生成文本: {cloud_text}")
        print(f"Cloud时间: {cloud_time:.4f}s")
        print(f"Cloud TTFT: {cloud_metrics.get('ttft', 'N/A'):.4f}s")
        print(f"Cloud OTPS: {cloud_metrics.get('otps', 'N/A'):.2f} tokens/s")
        
        # Speculative Decoding
        print("\n--- Speculative Decoding ---")
        start_time = time.time()
        spec_text, spec_metrics = spec_decoding.generate(
            test_data['audio_waveform'],
            test_data['prompt'],
            max_new_tokens=32
        )
        spec_time = time.time() - start_time
        
        print(f"Spec生成文本: {spec_text}")
        print(f"Spec时间: {spec_time:.4f}s")
        print(f"Spec输出tokens: {spec_metrics.get('output_tokens', 'N/A')}")
        print(f"Spec Cloud调用: {spec_metrics.get('cloud_calls', 'N/A')}")
        acceptance_rate = spec_metrics.get('acceptance_rate', 0)
        correction_rate = spec_metrics.get('correction_rate', 0)
        print(f"Spec接受率: {acceptance_rate:.2%}" if isinstance(acceptance_rate, (int, float)) else f"Spec接受率: {acceptance_rate}")
        print(f"Spec纠正率: {correction_rate:.2%}" if isinstance(correction_rate, (int, float)) else f"Spec纠正率: {correction_rate}")
        
        # 对比分析
        print(f"\n📊 对比分析:")
        print(f"  - 时间对比: Edge({edge_time:.4f}s) vs Cloud({cloud_time:.4f}s) vs Spec({spec_time:.4f}s)")
        print(f"  - TTFT对比: Edge({edge_metrics.get('ttft', 0):.4f}s) vs Cloud({cloud_metrics.get('ttft', 0):.4f}s)")
        print(f"  - OTPS对比: Edge({edge_metrics.get('otps', 0):.2f}) vs Cloud({cloud_metrics.get('otps', 0):.2f})")
        print(f"  - 文本长度: Edge({len(edge_text)}) vs Cloud({len(cloud_text)}) vs Spec({len(spec_text)})")
        
        # Speculative Decoding详细分析
        print(f"\n🔍 Speculative Decoding详细分析:")
        print(f"  - Cloud调用次数: {spec_metrics.get('cloud_calls', 0)}")
        print(f"  - 总draft tokens: {spec_metrics.get('total_draft_tokens', 0)}")
        print(f"  - 接受tokens: {spec_metrics.get('total_accepted_tokens', 0)}")
        print(f"  - 纠正次数: {spec_metrics.get('total_corrections', 0)}")
        print(f"  - 接受率: {spec_metrics.get('acceptance_rate', 0):.2%}")
        print(f"  - 纠正率: {spec_metrics.get('correction_rate', 0):.2%}")
        
        # 性能评估
        if spec_time > cloud_time * 1.2:  # 如果比Cloud慢20%以上
            print(f"  ⚠️  性能警告: Speculative Decoding比Cloud慢 {((spec_time/cloud_time-1)*100):.1f}%")
        if spec_metrics.get('cloud_calls', 0) > 5:  # 如果Cloud调用过多
            print(f"  ⚠️  效率警告: Cloud调用次数过多 ({spec_metrics.get('cloud_calls', 0)}次)")
        if spec_metrics.get('acceptance_rate', 0) < 0.7:  # 如果接受率过低
            print(f"  ⚠️  质量警告: 接受率过低 ({spec_metrics.get('acceptance_rate', 0):.2%})")
        
        return True
        
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 开始Speculative Decoding测试")
    
    # 全局初始化模型
    print("🔧 初始化模型...")
    edge_model = EdgeModel(device="cuda", dtype="float16")
    cloud_model = CloudModel(device="cuda", dtype="float16")
    print("✅ 模型初始化完成")
    
    # 只进行一次完整的对比测试
    success = compare_with_baselines(edge_model, cloud_model)
    
    if success:
        print("\n🎉 所有测试通过! Speculative Decoding工作正常。")
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")
        sys.exit(1)
