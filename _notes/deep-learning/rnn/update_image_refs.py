#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新03笔记中的图片引用为语义化命名
补充图片后运行此脚本
"""

import os

# 图片名称映射
replacements = {
    'image-23.png': 'vocab_frequency_stats.png',
    'image-24.png': 'word_frequency_distribution.png',
    'image-26.png': 'bigram_frequency.png',
    'image-27.png': 'trigram_frequency.png',
    'image-28.png': 'ngram_comparison.png',
    'batch_logic_clear-1.png': 'batch_processing_logic.png',
    'seq_partition.png': 'sequential_partition_diagram.png'
}

filename = '03-语言模型和数据集.md'

print(f"📝 正在更新 {filename} 的图片引用...\n")

# 读取文件
with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换引用
for old_name, new_name in replacements.items():
    old_ref = f'figures/{old_name}'
    new_ref = f'figures/{new_name}'
    if old_ref in content:
        content = content.replace(old_ref, new_ref)
        print(f"✓ {old_name} → {new_name}")
    else:
        print(f"ℹ 未找到引用: {old_name}")

# 写回文件
with open(filename, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\n✅ 图片引用更新完成！")
print(f"\n请确认以下图片文件存在于 figures/ 目录：")
for new_name in replacements.values():
    filepath = os.path.join('figures', new_name)
    if os.path.exists(filepath):
        print(f"  ✓ {new_name}")
    else:
        print(f"  ✗ {new_name} (缺失)")
