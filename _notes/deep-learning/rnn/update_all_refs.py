#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新所有markdown文件中的图片引用
"""

import os
import re

# 图片引用更新映射
replacements = {
    'image-23.png': 'vocab_frequency_stats.png',
    'image-24.png': 'word_frequency_distribution.png',
    'image-25.png': 'zipf_law_analysis.png',
    'image-26.png': 'bigram_frequency.png',
    'image-27.png': 'trigram_frequency.png',
    'image-28.png': 'ngram_comparison.png',
    'image-29.png': 'batch_processing_example.png',
    'image-30.png': 'sequence_sampling_method.png',
    'image-31.png': 'random_sampling.png',
    'image-32.png': 'sequential_sampling.png',
    'image-33.png': 'data_iterator_output1.png',
    'image-34.png': 'data_iterator_output2.png',
    'image-35.png': 'data_iterator_output3.png',
    'image-36.png': 'batch_size_comparison.png',
    'image-37.png': 'training_result_sequential.png',
    'image-38.png': 'training_result_random.png',
}

# 要更新的文件列表
markdown_files = [
    '01-序列建模基础.md',
    '02-文本数据处理.md',
    '03-语言模型和数据集.md',
    '04-循环神经网络从0开始实现.md'
]

print("📝 开始更新markdown文件中的图片引用...\n")

for filename in markdown_files:
    if not os.path.exists(filename):
        print(f"⚠️  文件不存在: {filename}")
        continue
    
    print(f"处理: {filename}")
    
    # 读取文件
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    update_count = 0
    
    # 替换所有图片引用
    for old_name, new_name in replacements.items():
        old_ref = f'figures/{old_name}'
        new_ref = f'figures/{new_name}'
        
        if old_ref in content:
            content = content.replace(old_ref, new_ref)
            update_count += 1
            print(f"  ✓ {old_name} → {new_name}")
    
    # 如果有更新，写回文件
    if content != original_content:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  💾 已保存 ({update_count} 处更新)\n")
    else:
        print(f"  ℹ️  无需更新\n")

print("="*60)
print("✅ 所有markdown文件的图片引用已更新！")
print("="*60)

# 验证图片文件是否存在
print("\n🔍 验证图片文件...")
figures_dir = 'figures'
missing_images = []

for new_name in replacements.values():
    filepath = os.path.join(figures_dir, new_name)
    if not os.path.exists(filepath):
        missing_images.append(new_name)

if missing_images:
    print(f"\n⚠️  以下图片文件不存在 ({len(missing_images)} 个):")
    for img in missing_images:
        print(f"   - {img}")
else:
    print("\n✅ 所有引用的图片文件都存在！")
