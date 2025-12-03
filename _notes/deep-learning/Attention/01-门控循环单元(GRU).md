---
layout: note_with_toc
title: 1. 注意力提示
description: Gated Recurrent Unit - advanced RNN architecture with reset and update gates
category: Deep Learning
subcategory: Advanced RNN
tags: [RNN, GRU, Gated Networks, Deep Learning, Neural Networks]
permalink: /notes/gated-recurrent-unit/
redirect_from:
  - /notes/门控循环单元（GRU）/
  - /notes/sequence-modeling-basics/
---

# 1. 注意力提示

注意力是一种稀缺资源，我的理解就是现在对你要做的事情进行分工分，谁是第一。

例子 1：视频或音乐 APP

你不付钱 → 用注意力看广告

你付钱 → 买的是“不被广告打扰的注意力空间”

🎮 例子 2：网络游戏

你花注意力打怪、刷装备 → 帮游戏变得有活力，吸引更多玩家

你花钱 → 买到了节省注意力的捷径（快速变强）

注意力不是免费的 —— 你要么花时间，要么花钱。


人类的感官每秒都在处理大量信息，但是大脑无法全部处理，在进化的过程中，能自动筛选和聚焦重要信息。


# 2. 注意力基本分类

来自心理学家威廉·詹姆斯的框架（19世纪90年代）：
注意力有两个来源：

 1. 非自主性注意（bottom-up，自下而上）

由外界刺激的突显性引起，是自动发生的。


桌上有报纸、论文、笔记本、书都很普通

但一个红色的杯子特别显眼
👉 你的注意力就会被它吸引
这是 不由自主的。

 2. 自主性注意（top-down，自上而下）

由人的目标、意愿、任务驱动，是主动控制的。


喝完咖啡，你想读书

于是你主动把视线移到书上
👉 这是你的意图和任务目标决定的注意力方向
这种注意力更稳定、强烈、专注。

# 3. Q K V

自主性和非自主性的注意力提示解释其基本及机制，通过神经网络去设计注意力的机制的框架。

非自主提示，这里简单地使用参数化的全连接层，甚至是非参数化的最大汇聚层或平均汇聚层,就像全连接层，最大池化，平均池化。