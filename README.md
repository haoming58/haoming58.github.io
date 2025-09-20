# Haoming 的个人网站

基于 Jekyll 构建的个人技术博客和知识分享平台，采用极简主义设计风格。

## 🌟 网站特色

- **极简设计**: 采用 Jekyll-YAMT 主题的简洁风格
- **响应式布局**: 完美适配桌面端和移动端
- **模块化配置**: 灵活的功能开关和自定义选项
- **代码高亮**: 支持多种编程语言的语法高亮
- **SEO优化**: 完整的搜索引擎优化支持
- **快速加载**: 优化的样式和资源加载

## 📁 项目结构

```
haoming58.github.io/
├── _config.yml              # 网站配置文件
├── _data/                   # 数据配置文件
│   ├── settings.yml         # 功能开关配置
│   └── social.yml           # 社交链接配置
├── _includes/               # 可重用组件
│   ├── head.html           # 页面头部
│   ├── header.html         # 网站头部
│   ├── footer.html         # 网站底部
│   ├── pages.html          # 导航页面
│   ├── social.html         # 社交链接
│   └── reading_time.html   # 阅读时间计算
├── _layouts/               # 页面布局模板
│   ├── default.html        # 默认布局
│   └── home.html           # 首页布局
├── _sass/                  # Sass样式文件
│   ├── base.scss           # 基础样式
│   ├── typography.scss     # 字体排版
│   ├── header.scss         # 头部样式
│   ├── home.scss           # 首页样式
│   ├── post.scss           # 文章样式
│   ├── footer.scss         # 底部样式
│   ├── paginator.scss      # 分页样式
│   ├── scrollbar.scss      # 滚动条样式
│   ├── mobile.scss         # 移动端样式
│   └── syntax.scss         # 代码高亮样式
├── _posts/                 # 博客文章
├── notes/                  # 学习笔记
│   ├── antenna/            # 天线设计笔记
│   ├── coding/             # 编程技术笔记
│   └── thoughts/           # 思考感悟
├── assets/                 # 静态资源
│   ├── css/                # 样式文件
│   └── images/             # 图片资源
├── about.markdown          # 关于页面
├── index.markdown          # 首页
└── README.md               # 项目说明
```

## 🚀 快速开始

### 1. 环境要求

- Ruby 2.7 或更高版本
- Jekyll 4.0 或更高版本
- Bundler

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/haoming58/haoming58.github.io.git
cd haoming58.github.io

# 安装依赖
bundle install
```

### 3. 本地开发

```bash
# 启动本地服务器
bundle exec jekyll serve

# 在浏览器中访问
# http://localhost:4000
```

### 4. 构建部署

```bash
# 构建静态文件
bundle exec jekyll build

# 部署到 GitHub Pages
git add .
git commit -m "Update website"
git push origin main
```

## ⚙️ 配置说明

### 网站基本信息

编辑 `_config.yml` 文件：

```yaml
# 网站基本信息
title: "你的网站标题"
description: "网站描述"
url: "https://your-username.github.io"
author:
  name: "你的姓名"
  email: "your.email@example.com"
  url: "https://your-username.github.io"
```

### 功能开关

编辑 `_data/settings.yml` 文件：

```yaml
# 功能开关
subtitle:
  active: true
  text: "你的副标题"

nav:
  pages:
    active: true
  social:
    active: true

featured-img:
  active: true

date-under-post:
  active: true

categories:
  active: true
```

### 社交链接

编辑 `_data/social.yml` 文件：

```yaml
social:
  - {icon: 'github', type: "brand", link: 'https://github.com/your-username'}
  - {icon: 'twitter', type: "brand", link: 'https://twitter.com/your-username'}
  - {icon: 'linkedin', type: "brand", link: 'https://linkedin.com/in/your-username'}
  - {icon: 'envelope', type: "solid", link: 'mailto:your.email@example.com'}
  - {icon: 'rss', type: "solid", link: "/feed.xml"}
```

## 📝 内容管理

### 添加博客文章

在 `_posts/` 目录下创建新的 Markdown 文件：

```markdown
---
layout: post
title: "文章标题"
date: 2024-01-15 10:00:00 +0800
categories: [技术分享]
tags: [JavaScript, React, 前端开发]
image: /assets/images/example.jpg
description: "文章描述"
---

# 文章标题

文章内容...
```

### 添加学习笔记

在 `notes/` 目录下的相应分类中创建新文件：

```markdown
---
layout: default
title: "笔记标题"
categories: [coding]
date: 2024-01-15
summary: "笔记摘要"
---

# 笔记标题

笔记内容...
```

### 支持的分类

- `coding`: 编程技术
- `antenna`: 天线设计
- `thoughts`: 思考感悟

## 🎨 样式定制

### 颜色主题

编辑 `_sass/base.scss` 文件中的颜色变量：

```scss
// 主色调
$primary-color: #2563eb;
$secondary-color: #64748b;
$accent-color: #f59e0b;
$base: #a2a2a2;
$light-grey: #ebebeb;
```

### 字体设置

编辑 `_sass/typography.scss` 文件：

```scss
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 16px;
  line-height: 1.7;
}
```

## 📱 响应式设计

网站采用响应式设计，自动适配不同设备：

- **桌面端**: 宽度 > 768px
- **平板端**: 宽度 480px - 768px
- **移动端**: 宽度 < 480px

## 🔧 功能特性

### 代码高亮

支持多种编程语言的语法高亮：

```javascript
// JavaScript 示例
function example() {
    console.log("Hello, World!");
}
```

```python
# Python 示例
def example():
    print("Hello, World!")
```

### 数学公式

支持 LaTeX 数学公式（需要启用 MathJAX）：

```latex
$$E = mc^2$$
```

### 目录生成

在文章中使用 `{:toc}` 自动生成目录。

### 阅读时间

自动计算文章阅读时间。

## 📊 SEO 优化

- 自动生成 sitemap.xml
- 支持 Open Graph 标签
- 支持 Twitter Cards
- 自动生成 RSS 订阅

## 🚀 部署

### GitHub Pages

1. 将代码推送到 GitHub 仓库
2. 在仓库设置中启用 GitHub Pages
3. 选择源分支（通常是 main）
4. 网站将自动部署到 `https://your-username.github.io`

### 自定义域名

1. 在仓库根目录创建 `CNAME` 文件
2. 在文件中写入你的域名
3. 在域名服务商处配置 DNS 记录

## 📚 学习资源

- [Jekyll 官方文档](https://jekyllrb.com/docs/)
- [Markdown 语法指南](https://www.markdownguide.org/)
- [Sass 官方文档](https://sass-lang.com/documentation)
- [Font Awesome 图标库](https://fontawesome.com/)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

本项目基于 MIT 许可证开源。

## 📞 联系方式

- **GitHub**: [haoming58](https://github.com/haoming58)
- **邮箱**: your.email@example.com
- **网站**: https://haoming58.github.io

---

*让我们一起学习，一起成长，用技术创造更美好的世界！*
