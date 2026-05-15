---
layout: note_with_toc
title: 3. 从头实现 GPT 模型
description: 搭建 GPT-2 (124M)：LayerNorm、GELU、FeedForward、TransformerBlock、文本生成与损失计算
category: LLM
subcategory: GPT Implementation
tags: [LLM, GPT, GPT-2, Transformer, GELU, LayerNorm, PyTorch, TextGeneration]
permalink: /notes/gpt-implementation/
---

# LLM 从零实现 — 三本笔记总览流程图

```
原始文本 (the-verdict.txt)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  📓 Tokenization.ipynb  【数据预处理与词表构建】             │
│                                                             │
│  ① 正则分词                                                 │
│     文本 ──regex split──▶ token 列表                        │
│                                                             │
│  ② 构建词表                                                  │
│     token 列表 ──sorted+enumerate──▶ {word: id} 词典        │
│                                                             │
│  ③ SimpleTokenizerV1 / V2                                  │
│     encode: 文本 ──▶ [id, id, ...]                          │
│     decode: [id, id, ...] ──▶ 文本                          │
│     V2 新增: 未知词 → <|unk|> 处理                          │
│                                                             │
│  ④ BPE 分词 (tiktoken / GPT-2)                              │
│     50257 词汇量, 字节级 BPE, 处理任意字符                  │
│                                                             │
│  ⑤ GPTDatasetV1 + DataLoader                               │
│     滑动窗口切片 ──▶ (input_ids, target_ids) 对             │
│     [B, seq_len] 批次张量，供模型训练使用                   │
└──────────────────────┬──────────────────────────────────────┘
                       │  Token ID 序列  [Batch, Seq]
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  📓 Self-attention.ipynb  【注意力机制】                     │
│                                                             │
│  ① 简单点积注意力                                            │
│     attn_score = q · kᵀ  →  softmax  →  权重求和           │
│                                                             │
│  ② SelfAttention_v1 (nn.Parameter)                         │
│     W_Q / W_K / W_V 手动参数, 完整 QKV 流程                │
│                                                             │
│  ③ SelfAttention_v2 (nn.Linear)                            │
│     用 Linear 替代 Parameter, 自动管理偏置                  │
│                                                             │
│  ④ CausalAttention                                         │
│     上三角 mask ──▶ 未来词得分 = -∞                        │
│     保证 GPT 只能看到当前及之前的词                         │
│                                                             │
│  ⑤ MultiHeadAttentionWrapper (朴素多头)                    │
│     n 个独立 CausalAttention ──concat──▶ out_proj          │
│                                                             │
│  ⑥ MultiHeadAttention (高效多头)                           │
│     单次 QKV 投影 ──reshape/transpose──▶ 多头并行计算      │
│     输出: [B, S, 768]，维度不变                             │
└──────────────────────┬──────────────────────────────────────┘
                       │  上下文向量  [Batch, Seq, 768]
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  📓 GPT.ipynb  【完整模型搭建与文本生成】                   │
│                                                             │
│  模块一  DummyGPT 占位符架构  ── 先跑通维度                 │
│  模块二  LayerNorm            ── 特征维度归一化              │
│  模块三  GELU 激活            ── 平滑非线性, 防神经元死亡   │
│  模块四  FeedForward          ── 768→3072→768, 提取特征     │
│  模块五  残差连接             ── x = x + F(x), 防梯度消失   │
│  模块六  Transformer Block    ── Attention + FFN + 2×残差   │
│  模块七  GPTModel (124M)      ── 嵌入 + 12×Block + 输出头  │
│  模块八  文本生成             ── 贪心自回归解码              │
│  模块九  损失计算             ── 交叉熵 = 负对数似然        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
          生成文本 / 训练 Loss → 反向传播 → 参数更新
```

```python
import torch          # 张量运算与自动微分
import torch.nn as nn  # 神经网络层（Embedding, Linear, Dropout…）
```

```python
# GPT-2 Small（124M 参数）超参数配置
GPT_CONFIG_124M = {
    "vocab_size":       50257,  # BPE 词表大小（tiktoken GPT-2 编码）
    "context_length":   1024,   # 最大上下文长度（位置嵌入的行数）
    "emb_dim":          768,    # 嵌入维度（所有向量的统一宽度）
    "n_heads":          12,     # 多头注意力的头数（每头 64 维）
    "n_layers":         12,     # Transformer 块的堆叠层数
    "drop_rate":        0.1,    # Dropout 概率（训练防过拟合，推理时关闭）
    "qkv_bias":         False   # QKV 投影是否带偏置（GPT-2 原版为 False）
}
```

## 模块一：占位符 GPT 架构

用 `Dummy*` 占位符组件先搭出完整数据流，验证从 token ID 到 logits 的张量维度正确，再逐步替换为真实实现。

数据流：`token ID [B,S]` → 词嵌入+位置嵌入 → `[B,S,768]` → Transformer×12 → LayerNorm → `[B,S,50257]` logits

假设我们给这个 GPT 喂了 1 句话，内容是 "Your journey starts"

输入： "Your journey starts"

变成 ID： [500, 8200, 120]

张量化 (in_idx)： 形状是 [1, 3]（Batch=1, 长度=3）。

tok_embeds = self.tok_emb(in_idx)

动作： 模型从那个 ,50257 ,768 的巨大“零件库”里，把第 500、8200、120 这三行抠出来。
维度： [1, 3] $\rightarrow$ [1, 3, 768]
理解： 现在的每一个词，已经由一个编号，变成了一个包含 768 个浮点数的长向量。

pos_embeds = self.pos_emb(torch.arange(3))

动作： 模型从另一个 $1024 \times 768$ 的“位置库”里，抠出前 3 行（第 0, 1, 2 行）。

维度： [3, 768]

理解： 这代表了“第一个位置”、“第二个位置”、“第三个位置”的坐标信息。

x = tok_embeds + pos_embeds

计算： [1, 3, 768] + [3, 768]

维度： 依然是 [1, 3, 768]

物理意义： 核心！ 这不是矩阵乘法，而是对应元素相加。比如词向量的第一个数加上坐标向量的第一个数。

结果： 此时的 x 既包含了这个词“是什么意思”，也包含了它“在句子的什么位置”

logits = self.out_head(x)

动作： 这是最惊人的跨越。每一个词位置的 768 个特征，被强制映射回词表大小。

计算： [1, 3, 768] $\times$ [768, 50257]

维度： [1, 3, 50257]理解： * 这一步产生了一个巨大的评分表。
 

```python
class DummyGPTModel(nn.Module):
    """占位符 GPT 模型：用真实的嵌入层 + 占位 Transformer/LayerNorm 跑通完整数据流。"""

    def __init__(self, cfg):
        super().__init__()
        # 词嵌入：将 token ID 映射为 768 维向量，形状 [50257, 768]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 位置嵌入：为每个位置（0~1023）提供坐标向量，形状 [1024, 768]
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 12 个占位 Transformer 块（原封不动传递输入，仅验证维度）
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        # 输出头：将 768 维特征投影回词表大小 [768→50257]，得到 logits
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # in_idx: [batch_size, seq_len]，每个值是 token 的整数 ID
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)                                   # [B, S, 768]
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # [S, 768]
        x = tok_embeds + pos_embeds    # 语义信息 + 位置信息，广播相加 [B, S, 768]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)      # [B, S, 50257]
        return logits
```

```python
class DummyTransformerBlock(nn.Module):
    """占位符 Transformer 块：不做任何处理，仅透传输入，用于验证维度。
    真实实现见模块六 TransformerBlock（含 MultiHeadAttention + FeedForward + 残差）。
    """
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x  # 维度 [B, S, 768] 不变
```

```python
class DummyLayerNorm(nn.Module):
    """占位符 LayerNorm：接受与官方 nn.LayerNorm 相同的参数（防止报错），但不做归一化。
    真实实现见模块二 LayerNorm 类。
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x  # 原样返回，不做任何处理
```

```python
import torch

torch.manual_seed(123)

# 模拟输入：2 条序列，各含 4 个 token ID（数字代表词表中的编号）
batch = torch.tensor([
    [15, 80, 20, 5],
    [90, 10,  3, 7]
])

# 用占位符模型验证完整数据流的维度是否正确
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)

print("输入形状 (Batch, Seq):", batch.shape)
print("输出形状 (Batch, Seq, Vocab):", logits.shape)
```

    输入形状 (Batch, Seq): torch.Size([2, 4])
    输出形状 (Batch, Seq, Vocab): torch.Size([2, 4, 50257])
    

## 模块二：层归一化（Layer Normalization）

对每个 Token 的特征向量**沿特征维度**独立做归一化（均值≈0、方差≈1），再由可学习的 `scale(γ)`/`shift(β)` 恢复表达能力。  
与 Batch Norm 的区别：LayerNorm 不依赖 batch 大小，适合变长序列，且在推理时行为一致。

```python
import torch
import torch.nn as nn

torch.manual_seed(123)

# 构造演示数据：2个样本，各有5个特征，用于手动推导 LayerNorm 的计算过程
batch_example = torch.randn(2, 5)
```

```python
# 用一个简单的 Linear + ReLU 层产生中间激活值
# 注意：这里用 ReLU 是为了让部分输出为 0，方便观察归一化效果
layer = nn.Sequential(
    nn.Linear(5, 6),
    nn.ReLU()
)
out = layer(batch_example)  # [2, 6]
print(out)
```

    tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
            [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
           grad_fn=<ReluBackward0>)
    

```python
# 沿特征维度（dim=-1）计算每个样本的均值和方差
# keepdim=True 保持维度以便后续广播
mean = out.mean(dim=-1, keepdim=True)  # [2, 1]
var  = out.var(dim=-1, keepdim=True)   # [2, 1]
print("均值:\n", mean)
print("方差:\n", var)
```

    均值:
     tensor([[0.1324],
            [0.2170]], grad_fn=<MeanBackward1>)
    方差:
     tensor([[0.0231],
            [0.0398]], grad_fn=<VarBackward0>)
    

```python
# 手动归一化：(x - mean) / std，结果应使均值≈0、方差≈1
out_norm = (out - mean) / torch.sqrt(var)

mean = out_norm.mean(dim=-1, keepdim=True)
var  = out_norm.var(dim=-1, keepdim=True)

print("归一化后的输出:\n", out_norm)
print("归一化后均值:\n", mean)
print("归一化后方差:\n", var)
```

    归一化后的输出:
     tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
            [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]],
           grad_fn=<DivBackward0>)
    归一化后均值:
     tensor([[    0.0000],
            [    0.0000]], grad_fn=<MeanBackward1>)
    归一化后方差:
     tensor([[1.0000],
            [1.0000]], grad_fn=<VarBackward0>)
    

```python
# 关闭科学计数法，让极小值（如 9.93e-09）显示为 0.0000
torch.set_printoptions(sci_mode=False)
print("均值（接近 0）:\n", mean)
print("方差（等于 1）:\n", var)
```

    均值（接近 0）:
     tensor([[    0.0000],
            [    0.0000]], grad_fn=<MeanBackward1>)
    方差（等于 1）:
     tensor([[1.0000],
            [1.0000]], grad_fn=<VarBackward0>)
    

```python
class LayerNorm(nn.Module):
    """层归一化：沿特征维度（最后一维）对每个 Token 独立归一化。
    
    公式：y = scale · (x - mean) / sqrt(var + eps) + shift
    - scale (γ): 可学习缩放，初始化为全 1
    - shift (β): 可学习平移，初始化为全 0
    - eps: 防止除零的极小值（1e-5）
    
    归一化后均值≈0、方差≈1，训练更稳定；scale/shift 让模型按需恢复分布。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习 γ
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习 β

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False：用 N 而非 N-1 做分母（偏置估计），与 PyTorch 官方 LayerNorm 一致
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

```python
# 验证自定义 LayerNorm：用 5 维特征的 batch_example 测试
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)  # [2, 5]

mean = out_ln.mean(dim=-1, keepdim=True)
var  = out_ln.var(dim=-1, unbiased=False, keepdim=True)

# 归一化结果应使每行均值≈0，方差≈1
print("Mean:\n", mean)
print("Variance:\n", var)
```

    Mean:
     tensor([[    -0.0000],
            [     0.0000]], grad_fn=<MeanBackward1>)
    Variance:
     tensor([[1.0000],
            [1.0000]], grad_fn=<VarBackward0>)
    

## 模块三：GELU 激活函数

与 ReLU 相比，GELU 在负数区保留微弱梯度（平滑过渡而非硬截断），可避免"神经元死亡"问题，是 GPT 系列的标准激活函数。  
公式：`GELU(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`

```python
class GELU(nn.Module):
    """高斯误差线性单元激活函数（GPT 系列标准）。
    
    公式（tanh 近似版）：
        GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
    
    与 ReLU 的区别：负数区不硬截断，保留微弱梯度，避免"死亡神经元"。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))
```

```python
import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU()

# 在 [-3, 3] 区间内对比 GELU 和 ReLU 的激活曲线
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()
```

         10 for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
         11     plt.subplot(1, 2, i)
         13     plt.title(f"{label} activation")
         14     plt.xlabel("x")
    

    File C:\envs\d2l\lib\site-packages\matplotlib\pyplot.py:2988, in plot(scalex, scaley, data, *args, **kwargs)
       2986 @_copy_docstring_and_deprecators(Axes.plot)
       2987 def plot(*args, scalex=True, scaley=True, data=None, **kwargs):
       2989         *args, scalex=scalex, scaley=scaley,
       2990         **({"data": data} if data is not None else {}), **kwargs)
    

    File C:\envs\d2l\lib\site-packages\matplotlib\axes\_axes.py:1607, in Axes.plot(self, scalex, scaley, data, *args, **kwargs)
       1605 lines = [*self._get_lines(*args, data=data, **kwargs)]
       1606 for line in lines:
       1608 self._request_autoscale_view(scalex=scalex, scaley=scaley)
       1609 return lines
    

    File C:\envs\d2l\lib\site-packages\matplotlib\axes\_base.py:2105, in _AxesBase.add_line(self, line)
       2102 if line.get_clip_path() is None:
       2103     line.set_clip_path(self.patch)
       2106 if not line.get_label():
       2107     line.set_label('_line%d' % len(self.lines))
    

    File C:\envs\d2l\lib\site-packages\matplotlib\axes\_base.py:2127, in _AxesBase._update_line_limits(self, line)
       2123 def _update_line_limits(self, line):
       2124     """
       2125     Figures out the data limit of the given line, updating self.dataLim.
       2126     """
       2128     if path.vertices.size == 0:
       2129         return
    

    File C:\envs\d2l\lib\site-packages\matplotlib\lines.py:1022, in Line2D.get_path(self)
       1017 """
       1018 Return the :class:`~matplotlib.path.Path` object associated
       1019 with this line.
       1020 """
       1021 if self._invalidy or self._invalidx:
       1023 return self._path
    

    File C:\envs\d2l\lib\site-packages\matplotlib\lines.py:663, in Line2D.recache(self, always)
        661 if always or self._invalidx:
        662     xconv = self.convert_xunits(self._xorig)
        664 else:
        665     x = self._x
    

![GELU vs ReLU]({{ '/assets/img/notes/LLM/3.GPT_19_1.png' | relative_url }})
    

```python
class FeedForward(nn.Module):
    """Transformer 块内的两层前馈网络（Position-wise FFN）。
    
    结构：Linear(768→3072) → GELU → Linear(3072→768)
    维度始终保持 [B, S, 768]，4× 扩展给模型足够的非线性表达空间。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 扩展：768 → 3072
            GELU(),                                          # 平滑非线性激活
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])   # 压缩：3072 → 768
        )

    def forward(self, x):
        return self.layers(x)
```

```python
# 验证 FeedForward 维度保持不变：[B, S, 768] → [B, S, 768]
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print("输出形状（应与输入一致）:", out.shape)  # torch.Size([2, 3, 768])
```

## 模块四：前馈网络（FeedForward）

每个 Transformer 块内的两层 MLP：  
`Linear(768→3072)` → `GELU` → `Linear(3072→768)`  
扩展4倍再压缩，给网络足够的"中间空间"提取非线性特征，维度始终保持 `[B, S, 768]`。

特性	Python list	PyTorch nn.ModuleList

存储层	可以存储	可以存储

权重注册	不支持（模型找不到参数）	支持（参数自动加入模型）

.to(device)	无效（内部层不会移动）	有效（内部层整体移动）

适用场景	存储普通的数字或字符串	存储需要训练的神经网络层

## 模块五：残差连接（Residual Connection）

通过 `x = x + F(x)` 构造捷径通路，让梯度可以绕过 F(x) 直接回传，是深层网络（12层+）能正常训练的关键。  
`use_shortcut=True` 时开启，要求输入输出形状相同。

```python
# 2. 定义深度残差网络
class DeepResidualNetwork(nn.Module):
    
    def __init__(self, layer_sizes, use_shortcut): 
        
        super().__init__()
        
        self.use_shortcut = use_shortcut # 确保能够使用这个，保证维度的一致性
        
        # ModuleList 确保 PyTorch 能追踪到这些层的参数
        
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1): # 这里有一个参数是 layer_block
            
            # 每一个 Block 包含：线性层 + 归一化 + 激活
            
            layer_block = nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                
                LayerNorm(layer_sizes[i+1]), # 在激活前归一化
                
                nn.GELU()
            )
            
            self.layers.append(layer_block) 

            # 最终加入到这个，叫 做 layers 的东西

    def forward(self, x):
        
        for layer in self.layers:
            
            layer_output = layer(x)
            
            # 残差连接逻辑
            if self.use_shortcut and x.shape == layer_output.shape:
                
                x = x + layer_output # “捷径”：保留原始信息流
            else:
                x = layer_output # 传统覆盖逻辑
                
        return x

# --- 测试运行 ---
# 设置维度：输入 768，中间层全部保持 768 维度以启用残差

dims = [768, 768, 768, 768, 768] # 每一层的输出维度

model = DeepResidualNetwork(layer_sizes=dims, use_shortcut=True)

# 模拟输入：Batch=2, Seq=10, Dim=768
example_input = torch.randn(2, 10, 768)
output = model(example_input)

print(f"输入形状: {example_input.shape}")
print(f"输出形状: {output.shape}")
```

```python
def print_gradients(model, x):
    
    # 1. 前向传播
    
    output = model(x)
    
    # 2. 定义目标值 (Target)
    
    # 假设输出形状是 (Batch, Seq, Dim)，我们需要将其展平或取均值来计算 Loss

    # 因为，会一次性地计算出很多的值和误差，到底改听谁的，因此，就平均，听取大家的意见。
    
    # 这里简单地对所有输出取平均值，目标设为 0
    
    target = torch.zeros_like(output)  # 这里生成一个和之类类似的张量，当作容易使用。
    
    # 3. 实例化损失函数并计算
    
    criterion = nn.MSELoss()
    
    loss = criterion(output, target)

    # 4. 反向传播：这是产生梯度的关键一步！
    
    model.zero_grad() # 清除之前的旧梯度
    
    loss.backward()

    # 5. 遍历并打印每一层的梯度
    print(f"{'层名称':<30} | {'梯度范数 (Gradient Norm)':<20}")
    
    print("-" * 55)
    
    for name, param in model.named_parameters():
        
        if param.grad is not None:
            # 使用梯度范数（Norm）来代表梯度的大小
            grad_norm = param.grad.norm().item() # 这里使用了一个叫范数的东西
            print(f"{name:<30} | {grad_norm:.6f}")
        else:
            print(f"{name:<30} | None (无梯度)")

```

```python
import torch
import torch.nn as nn

# --- 1. 定义你的残差网络 ---
class DeepResidualNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut): 
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1): 
            # 优化点：标准的 Transformer/ResNet 模块写法
            # 保证 Block 的最后一层是线性层，这样输出范围无限制，能完美逼近任何目标
            layer_block = nn.Sequential(
                nn.LayerNorm(layer_sizes[i]),                 # 1. 归一化
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),  # 2. 线性映射
                nn.GELU(),                                    # 3. 激活函数（空间扭曲）
                nn.Linear(layer_sizes[i+1], layer_sizes[i+1]) # 4. 再次线性映射作为结尾
            )
            self.layers.append(layer_block) 

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            
            # 残差连接逻辑
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output # “捷径”
            else:
                x = layer_output     # 覆盖
        return x

# --- 2. 见证奇迹的训练循环 ---
def train_and_watch(model, x):
    # 设定目标：我们强行要求网络把输出全变成 0
    target = torch.zeros_like(x)
    
    # 均方误差损失函数
    criterion = nn.MSELoss()
    
    # 请来执行官（优化器），学习率设为 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("🚀 训练开始！请盯紧 Loss 的数值变化：\n")
    print(f"{'训练轮数 (Epoch)':<15} | {'当前误差 (Loss)':<20}")
    print("-" * 40)

    # 循环训练 50 次
    for epoch in range(50):
        
        # 第 1 步：前向传播（盲人瞎猜）
        output = model(x)
        
        # 第 2 步：计算误差（离目标 0 有多远）
        loss = criterion(output, target)
        
        # 第 3 步：清空旧账，反向传播（查明责任，生成梯度）
        optimizer.zero_grad() 
        loss.backward()
        
        # 第 4 步：执行官发威！(根据梯度，真正动手修改所有的权重 W 和偏置 b)
        optimizer.step()
        
        # 打印当前这一轮的误差（为了不刷屏，第一轮必打印，后面每 5 轮打印一次）
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"第 {epoch + 1:<10} 轮 | {loss.item():.6f}")

# --- 3. 运行测试 ---
if __name__ == "__main__":
    # 设置维度：5 层网络，全都保持 768
    dims = [768, 768, 768, 768, 768]
    
    # 实例化模型，开启残差
    model = DeepResidualNetwork(dims, use_shortcut=True)
    
    # 模拟输入：2 个样本，序列长度 10，维度 768
    # 固定随机种子，保证每次生成的假数据一样，方便观察
    torch.manual_seed(42) 
    x = torch.randn(2, 10, 768)
    
    # 启动！
    train_and_watch(model, x)
```

## 模块六：Transformer 块（Transformer Block）

标准 Pre-Norm 结构（先归一化再送入子层）：  
```
x → LayerNorm → MultiHeadAttention → Dropout → +x（残差）
  → LayerNorm → FeedForward       → Dropout → +x（残差）
```
GPT-2 将 12 个这样的块串联（`n_layers=12`），逐层提炼语义表示。

```python
class TransformerBlock(nn.Module):
    
    def __init__(self, cfg):

        # cfg 自动的GPT参数
        
        super().__init__()
        
        # 1. 多头注意力机制（让词与词互相交流）
        
        # 注意：这里假设你已经有了一个 MultiHeadAttention 类
        
        self.att = MultiHeadAttention(
            
            d_in=cfg["emb_dim"],
            
            d_out=cfg["emb_dim"],
            
            context_length=cfg["context_length"],
            
            num_heads=cfg["n_heads"],
            
            dropout=cfg["drop_rate"],
            
            qkv_bias=cfg["qkv_bias"]
            
        )
        
        # 2. 前馈神经网络（也就是你之前写的那种 Linear -> GELU -> Linear）
         
        self.ff = FeedForward(cfg)
        
        # 3. 两个层归一化 (LayerNorm)
        
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        
        # 4. Dropout 层（用来防止过拟合，随机丢弃一些神经元）
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    
    def forward(self, x):
        # -------------------------------------------
        # 第一阶段：Attention (注意力分支) + 残差连接
        # -------------------------------------------
        shortcut = x                  # 记录原车 x (Shortcut connection for attention block)
        x = self.norm1(x)             # Pre-Norm 前置归一化
        x = self.att(x)               # 注意力机制提取特征
        x = self.drop_shortcut(x)     # 防止过拟合
        x = x + shortcut              # 残差相加 (Add the original input back)

        # -------------------------------------------
        # 第二阶段：Feed Forward (前馈分支) + 残差连接
        # -------------------------------------------
        shortcut = x                  # 再次记录目前的 x (Shortcut connection for feed forward block)
        x = self.norm2(x)             # 再次前置归一化
        x = self.ff(x)                # 经过你最熟悉的 Linear + GELU 处理
        x = self.drop_shortcut(x)     # 防止过拟合
        x = x + shortcut              # 再次残差相加 (Adds the original input back)
        
        return x

# ==========================================
# 补充：让你能跑通的依赖组件 (Dummy / 简化版实现)
# ==========================================

class FeedForward(nn.Module):
    """这不就是你刚刚改好的带有 GELU 的模块嘛！"""
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        # 大模型里，中间层的维度通常是输入维度的 4 倍
        hidden_dim = emb_dim * 4 
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """这里用一个简单的线性层占位，真实的注意力机制比这个复杂，涉及 Q, K, V 矩阵的运算"""
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias):
        
        super().__init__()
        
        self.dummy_layer = nn.Linear(d_in, d_out)
        
    def forward(self, x):
        # 真实的 Attention 会在这里计算注意力分数
        return self.dummy_layer(x)
```

```python
torch.manual_seed(123)
# 验证 TransformerBlock：[B, S, 768] → [B, S, 768]，维度不变
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
```

## 模块七：完整 GPT 模型

将所有组件组装为 GPT-2（124M 参数）：  
```
token ID [B,S] → tok_emb[50257,768] + pos_emb[1024,768]
              → Dropout
              → TransformerBlock × 12
              → LayerNorm
              → Linear(768→50257)   [= out_head]
              → logits [B, S, 50257]
```

```python
class GPTModel(nn.Module):
    
    def __init__(self, cfg):
        
        # 自动化导入配置
        
        super().__init__()
        
        # 词向量：我是谁？
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 位置向量：我在哪？
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # 12层连环加工车间
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # 最后质检
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        
        # 翻译官：把特征变回单词概率
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):

        # in_id 是 batch，seq，emd
        batch_size, seq_len = in_idx.shape
        
        device = in_idx.device
        
        # 1. 注入语义信息
        tok_embeds = self.tok_emb(in_idx)
        # 2. 注入时空信息 (0, 1, 2, ...)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=device))
        
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # 3. 经历所有 Transformer Blocks 的加工
        x = self.trf_blocks(x)
        
        # 4. 归一化输出
        x = self.final_norm(x)
        
        # 5. 生成 Logits（概率分布得分）
        logits = self.out_head(x)
        return logits
```

```python
torch.manual_seed(123)

# 重新定义 batch（与占位符测试一致），确保本 cell 独立可运行
batch = torch.tensor([
    [15, 80, 20, 5],
    [90, 10,  3, 7]
])

# 用真实 GPT 模型前向传播，验证输出形状
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)

print("Input batch:\n", batch)
print("Input batch.shape:", batch.shape)
print("Output shape:", out.shape)
```

```python
# 统计模型总参数量（包含所有可训练参数）
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
```

```python
# 词嵌入层与输出头共享同一组权重（Weight Tying），二者形状完全相同
print("Token embedding layer shape:", model.tok_emb.weight.shape)   # [50257, 768]
print("Output layer shape:", model.out_head.weight.shape)            # [50257, 768]
```

    Token embedding layer shape: torch.Size([50257, 768])
    Output layer shape: torch.Size([50257, 768])
    

```python
# Weight Tying：tok_emb 与 out_head 共享权重，减少 50257×768 ≈ 3860万参数
# 实际可训练参数 = 总参数 - out_head 参数
total_params_gpt2 = (
    total_params - sum(p.numel() for p in model.out_head.parameters())
)
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
```

```python
# 每个参数占 4 字节（float32），换算为 MB
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")
```

## 模块八：文本生成（Text Generation）

以**贪心解码**演示自回归生成：每步取 logits 中概率最高的词 ID 拼接到序列尾部，循环 `max_new_tokens` 次。  
注意：未经训练的模型生成结果是随机的，训练后才有意义的输出。

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    GPT 文本生成函数
    model: 你的 GPTModel 实例
    idx: 输入的初始词 ID 序列 [Batch, Seq]
    max_new_tokens: 想要生成多少个新词
    context_size: 模型的最大记忆长度 (cfg["context_length"])
    """
    for _ in range(max_new_tokens):
        
        # 1. 裁剪上下文：如果输入的词数超过了模型能记住的长度，只取最后一段
        idx_cond = idx[:, -context_size:]
        
        # 2. 获取预测结果 (Logits)
        with torch.no_grad(): # 生成时不需要计算梯度，节省显存
            logits = model(idx_cond)
        
        # 3. 核心步骤：只取最后一个时间步的预测得分 [Batch, Seq, Vocab] -> [Batch, Vocab]
        logits = logits[:, -1, :]
        
        # 4. 转换成概率 (可选，但在做采样时必选)
        probs = torch.softmax(logits, dim=-1)
        
        # 5. 找到概率最高的那一个词的索引 [Batch, 1]
        next_idx = torch.argmax(probs, dim=-1, keepdim=True)
        
        # 6. 将新生成的词 ID 拼接到原始序列后面
        idx = torch.cat((idx, next_idx), dim=1)
        
    return idx
```

```python
import torch
import tiktoken # 假设你使用的是 GPT-2 的分词器

# 1. 初始化分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 2. 定义你想对模型说的话
text = "Hello, I am"

# 3. 把文字变成数字 ID
encoded = tokenizer.encode(text)

# 4. 把数字变成 PyTorch 张量，并增加一个 Batch 维度 (变成 [[...]])
# 这就是你缺少的那个 'encoded_tensor'！
encoded_tensor = torch.tensor(encoded).unsqueeze(0) 

# 现在再运行你原来的代码，就不会报错了
model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor, # 现在它被定义好了
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

# 1. 把张量转回文字
# out.squeeze(0) 是把 [1, 10] 变成 [10]

# .tolist() 是把张量变成普通的 Python 列表

decoded_text = tokenizer.decode(out.squeeze(0).tolist())

# 2. 打印最终结果
print("生成的文字内容:")
print(decoded_text)
```

    生成的文字内容:
    Hello, I am drone Omni SSLmyra muc native
    

```python
def text_to_token_ids(text, tokenizer):
    
    encoded = tokenizer.encode(text)
    
    # 增加 Batch 维度，变成 [1, seq_len]
    
    return torch.tensor(encoded).unsqueeze(0)
    
    # 它把形状从 [seq_len] 变成了 [1, seq_len]

def token_ids_to_text(token_ids, tokenizer):
    
    # 用 .squeeze() 移除所有大小为 1 的维度，确保变成一维列表
    
    flat = token_ids.squeeze().tolist() 

    # 把 PyTorch 的张量格式转回 Python 普通列表，因为分词器的 decode 方法只认列表。
    
    return tokenizer.decode(flat)

# 确保模型在推理模式
model.eval()

start_context = "Every effort moves you"

# 确保你的 GPT_CONFIG_124M 和 model 已经定义好了

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=20,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

    Output text:
     Every effort moves you horizontRoomki laserutfucks Levels Nice Sir dentistvertingIKLG queer()PC Byzantine terrorist UEfol
    

## 模块九：损失计算（Cross-Entropy Loss）

语言模型的训练目标：最大化正确下一词的概率，等价于最小化**交叉熵损失**（负对数似然均值）。  
手动推导路径：`logits → softmax → 取正确词概率 → log → 取均值 → 取负号 = cross_entropy`

```python
# 构造两条样例输入序列（已经过 tiktoken 编码）
# "every effort moves" → [16833, 3626, 6100]
# "I really like"      → [40, 1107, 588]
inputs = torch.tensor([
    [16833, 3626, 6100],
    [40,    1107,  588]
])
```

```python
# 目标序列 = 输入右移 1 位（语言模型监督信号：预测下一个词）
# "effort moves you"   → [3626, 6100, 345]
# "really like them"   → [1107, 588, 11311]
targets = torch.tensor([
    [3626, 6100,   345],
    [1107,  588, 11311]
])
```

```python
# 前向传播获取 logits，再转为概率分布
# torch.no_grad() 关闭梯度计算，推理时节省显存
with torch.no_grad():
    logits = model(inputs)         # [2, 3, 50257]

probas = torch.softmax(logits, dim=-1)  # 对词表维度归一化，得到每个位置的概率分布
print("概率分布形状:", probas.shape)     # [2, 3, 50257]
```

    概率分布形状: torch.Size([2, 3, 50257])
    

```python
# 贪心预测：每个位置取概率最高的 token ID
token_ids = torch.argmax(probas, dim=-1, keepdim=True)  # [2, 3, 1]
print("预测 Token IDs:\n", token_ids)
```

    预测 Token IDs:
     tensor([[[25388],
             [41068],
             [49272]],
    
            [[ 2109],
             [ 3957],
             [30269]]])
    

```python
# 对比目标词与模型预测词（未训练模型预测结果是随机的）

print(f"目标批次 1: {token_ids_to_text(targets[0], tokenizer)}")

print(f"输出批次 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
```

    目标批次 1:  effort moves you
    输出批次 1:  compressedseq deterrence
    

看这个没有经过训练的，正确词的概率。

然后，看这个训练过后的概率是多少。

```python
# 手动推导交叉熵：从概率 -> 对数 -> 均值 -> 取负

# 1. 从概率分布中取出正确答案（targets）对应位置的概率值
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("文本 1 正确词概率:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("文本 2 正确词概率:", target_probas_2)

# 2. 对概率取对数
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print("对数概率:", log_probas)

# 3. 求均值
avg_log_probas = torch.mean(log_probas)
print("平均对数概率:", avg_log_probas)

# 4. 取负数得到交叉熵损失
neg_avg_log_probas = avg_log_probas * -1
print("交叉熵损失（手动推导）:", neg_avg_log_probas)

```

    文本 1 正确词概率: tensor([    0.0000,     0.0000,     0.0000])
    文本 2 正确词概率: tensor([    0.0000,     0.0000,     0.0000])
    对数概率: tensor([-12.2441, -12.0361, -12.5687, -11.2636, -10.6095, -11.4755])
    平均对数概率: tensor(-11.6996)
    交叉熵损失（手动推导）: tensor(11.6996)
    

```python
# PyTorch cross_entropy 要求输入为 [N, C]，目标为 [N]
# N = batch_size × seq_len = 2×3 = 6，C = 词表大小 50257
logits_flat = logits.flatten(0, 1)   # [2,3,50257] → [6,50257]
targets_flat = targets.flatten()     # [2,3]       → [6]

print("展平后的 logits:", logits_flat.shape)
print("展平后的目标:", targets_flat.shape)

# 一行等价于上面的手动推导过程：
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print("Cross-entropy loss:", loss)
```

    展平后的 logits: torch.Size([6, 50257])
    展平后的目标: torch.Size([6])
    Cross-entropy loss: tensor(11.6996)
    

上面手动推导的过程与 PyTorch 内置函数完全等价：

```python
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
```

`cross_entropy` 内部已包含 softmax + log + 取均值 + 取负号，直接输入原始 logits 即可。

## 模块十：计算训练和验证集的损失

```python
file_path = "the-verdict.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()  # 读取整个文本文件作为训练数据

```

```python
total_characters = len(text_data)

total_tokens = len(tokenizer.encode(text_data))

print("Characters:", total_characters)

print("Tokens:", total_tokens)

```

    Characters: 20479
    Tokens: 5145
    

```python

```
