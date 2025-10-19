#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整理新添加的图片
将image-XX.png重命名为语义化命名，并更新markdown引用
"""

import os
import shutil

# 图片重命名映射表
# 基于图片在markdown中的用途
image_mappings = {
    # 03-语言模型和数据集.md 中使用的图片
    'image-23.png': 'vocab_frequency_stats.png',           # 词频统计结果
    'image-24.png': 'word_frequency_distribution.png',     # 词频分布图
    'image-26.png': 'bigram_frequency.png',                # 二元语法词频
    'image-27.png': 'trigram_frequency.png',               # 三元语法词频
    'image-28.png': 'ngram_comparison.png',                # N-gram对比
    
    # 其他可能的图片（需要检查使用位置）
    'image-25.png': 'zipf_law_analysis.png',               # 可能是Zipf定律分析
    'image-29.png': 'batch_processing_example.png',        # 批处理示例
    'image-30.png': 'sequence_sampling_method.png',        # 序列采样方法
    'image-31.png': 'random_sampling.png',                 # 随机采样
    'image-32.png': 'sequential_sampling.png',             # 顺序采样
    'image-33.png': 'data_iterator_output1.png',           # 数据迭代器输出1
    'image-34.png': 'data_iterator_output2.png',           # 数据迭代器输出2
    'image-35.png': 'data_iterator_output3.png',           # 数据迭代器输出3
    'image-36.png': 'batch_size_comparison.png',           # 批次大小对比
    'image-37.png': 'training_result_sequential.png',      # 训练结果（顺序）
    'image-38.png': 'training_result_random.png',          # 训练结果（随机）
}

figures_dir = 'figures'

print("📦 开始整理新添加的图片...\n")
print(f"图片目录: {figures_dir}/\n")

# 统计
renamed_count = 0
skipped_count = 0
missing_count = 0

for old_name, new_name in image_mappings.items():
    old_path = os.path.join(figures_dir, old_name)
    new_path = os.path.join(figures_dir, new_name)
    
    if os.path.exists(old_path):
        if not os.path.exists(new_path):
            shutil.move(old_path, new_path)
            print(f"✓ {old_name:25s} → {new_name}")
            renamed_count += 1
        else:
            print(f"⚠ {old_name:25s} → {new_name} (目标已存在，跳过)")
            skipped_count += 1
    else:
        print(f"✗ {old_name:25s} (文件不存在)")
        missing_count += 1

print(f"\n" + "="*60)
print(f"重命名完成: {renamed_count} 个")
print(f"跳过: {skipped_count} 个")
print(f"缺失: {missing_count} 个")
print("="*60)

# 检查是否还有未整理的image-XX.png文件
remaining_images = [f for f in os.listdir(figures_dir) 
                   if f.startswith('image-') and f.endswith('.png')]

if remaining_images:
    print(f"\n⚠️  还有 {len(remaining_images)} 个未整理的图片:")
    for img in remaining_images:
        print(f"   - {img}")
else:
    print("\n✅ 所有image-XX.png文件已整理完成！")

print("\n📝 下一步: 运行 update_image_refs.py 更新markdown引用")
