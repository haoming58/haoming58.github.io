# Haoming 的个人博客

这是一个基于 Jekyll 构建的个人博客网站，具有现代化的设计和响应式布局。

## ✨ 功能特性

- 🎨 **现代化设计**: 采用渐变背景、卡片式布局和优雅的动画效果
- 📱 **响应式布局**: 完美适配桌面端、平板和移动设备
- 🧭 **导航系统**: 固定顶部导航栏，支持平滑滚动
- 🚀 **项目展示**: 精美的项目卡片展示，支持标签和链接
- 📝 **博客系统**: 基于 Markdown 的博客写作系统
- 🔍 **SEO 优化**: 针对搜索引擎优化的页面结构

## 🏗️ 技术架构

- **静态站点生成器**: Jekyll 4.4.1
- **主题**: 自定义设计，不依赖第三方主题
- **样式**: 纯 CSS3，支持现代浏览器特性
- **布局**: 响应式网格布局，使用 CSS Grid 和 Flexbox
- **图标**: 使用 Emoji 和 SVG 图标

## 📁 文件结构

```
haoming58.github.io/
├── _layouts/          # 页面布局模板
│   ├── home.html      # 首页布局
│   ├── projects.html  # 项目页面布局
│   └── default.html   # 默认布局
├── _posts/            # 博客文章
├── _site/             # 构建输出目录
├── index.markdown     # 首页内容
├── projects.md        # 项目页面
├── about.markdown     # 关于页面
├── _config.yml        # 网站配置
└── Gemfile           # Ruby 依赖管理
```

## 🚀 快速开始

### 环境要求

- Ruby 3.0+ 
- Jekyll 4.0+
- Bundler

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/haoming58/haoming58.github.io.git
cd haoming58.github.io
```

2. 安装依赖
```bash
bundle install
```

3. 启动本地服务器
```bash
bundle exec jekyll serve
```

4. 访问网站
打开浏览器访问 `http://localhost:4000`

### 构建部署

```bash
# 构建网站
bundle exec jekyll build

# 构建并启动服务器
bundle exec jekyll serve --watch
```

## 📝 内容管理

### 添加新文章

在 `_posts/` 目录下创建新的 Markdown 文件：

```markdown
---
layout: post
title: "文章标题"
date: 2025-01-XX
categories: [分类]
tags: [标签1, 标签2]
---

文章内容...
```

### 修改首页项目

编辑 `index.markdown` 文件中的 `featured_projects` 部分：

```yaml
featured_projects:
  - title: "项目标题"
    description: "项目描述"
    icon: "🚀"
    tags: ["标签1", "标签2"]
    link: "/projects#project-id"
```

### 自定义样式

所有样式都在 `_layouts/` 目录下的 HTML 文件中定义，可以直接修改 CSS 来自定义外观。

## 🎨 自定义配置

### 网站基本信息

编辑 `_config.yml` 文件：

```yaml
title: "你的网站标题"
description: "网站描述"
url: "你的网站URL"
author: "你的名字"
```

### 导航菜单

在布局文件中修改导航链接：

```html
<ul class="nav-links">
    <li><a href="/">主页</a></li>
    <li><a href="/projects">项目</a></li>
    <li><a href="/about">关于</a></li>
</ul>
```

## 🌟 特色功能

### 1. 渐变背景
首页使用现代化的渐变背景，营造视觉冲击力。

### 2. 项目卡片
精美的项目展示卡片，支持悬停动画效果。

### 3. 响应式设计
完美适配各种设备尺寸，提供最佳用户体验。

### 4. 平滑滚动
页面内导航支持平滑滚动效果。

## 📱 浏览器支持

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- **GitHub**: [haoming58](https://github.com/haoming58)
- **网站**: [haoming58.github.io](https://haoming58.github.io)

---

*用代码记录生活，用技术连接世界* 🚀
