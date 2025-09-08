# Haoming 的个人博客

_config.yml：网站的全局配置文件。你可以在这里设置网站标题、导航菜单、作者信息等。
Gemfile：定义了网站所需的 Ruby gem（插件和依赖），如 Jekyll、主题等。
_layouts：存放页面布局模板（如 default.html），决定所有页面的结构和样式。
_posts：存放博客文章，文件名格式为 YYYY-MM-DD-title.markdown。
index.markdown、about.markdown、projects.md：网站的主要页面内容。
assets/（如果有）：存放图片、CSS、JS 等静态资源。
其他文件如 404.html、favicon.ico.png：分别是 404 页面和网站图标

读取 _config.yml 配置。

加载 _layouts 提供的 HTML 结构。

将 .md 内容渲染成 HTML。

应用 Front Matter 中的参数。

使用 Gemfile 定义的插件扩展功能。

输出静态 HTML 页面。
## 📁 文件结构

```
haoming58.github.io/
├── _layouts/          # 页面布局模板
│   ├── home.html      # 首页布局
│   ├── about.html     # 个人介绍布局
│   ├── blog.html      # 博客列表布局
│   ├── post.html      # 单篇文章布局
│   └── default.html   # 默认布局（若使用）
├── _posts/            # 博客文章
├── _site/             # 构建输出目录
├── index.markdown     # 首页内容（前置信息）
├── about.markdown     # 关于页面
├── blog.md            # 博客列表入口
├── _config.yml        # 网站配置
└── Gemfile            # Ruby 依赖管理
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

### 调整首页文案

编辑 `index.markdown` 的前置信息：

```yaml
---
layout: home
title: 首页
hero_title: "你好，我是小铭"
hero_subtitle: "喜欢研究，分享和探讨知识的见解"
---
```

### 导航菜单

在 `_config.yml` 中配置导航（模板自动渲染）：

```yaml
navigation:
  - name: 主页
    link: /
  - name: 个人介绍
    link: /about
  - name: 博客
    link: /blog
```

### 自定义样式

所有样式都在 `
```

## 🧩 快速模板与操作指南

### 新增一篇博客（推荐）
在 `_posts/` 目录下新建文件，命名格式 `YYYY-MM-DD-title.md`：

```markdown
---
layout: post
title: "文章标题"
date: 2025-09-01
# 可选：tags 会显示在文章页、便于检索
tags: [学习, 笔记]
# 可选：excerpt 用于列表摘要（不写则自动截取）
excerpt: 这是一段自定义摘要，用于列表页展示。
---

这里是正文内容，支持标准 Markdown 语法。

- 支持代码块
- 支持图片与链接
- 支持表格
```

### 新增一个页面（如「联系我」「作品集」）
在仓库根目录新增 `contact.markdown`（或任意名字，注意 permalink）：

```markdown
---
layout: page
title: 联系我
permalink: /contact
---

这里写页面内容，支持 Markdown。
```

将新页面加入导航（可选）：编辑 `_config.yml`：

```yaml
navigation:
  - name: 主页
    link: /
  - name: 个人介绍
    link: /about
  - name: 博客
    link: /blog
  - name: 联系我
    link: /contact
```

若不存在 `page` 布局，可复制 `_layouts/about.html` 为 `_layouts/page.html` 并做简单改名与文案调整，然后在页面前置信息里使用 `layout: page`。

### 本地预览与发布
- 本地预览：
```bash
bundle install
bundle exec jekyll serve
```
浏览器访问 `http://127.0.0.1:4000`

- 构建发布（GitHub Pages 建议直接推送到主分支）：
```bash
bundle exec jekyll build
# 推送到 GitHub 后，Pages 会自动构建并发布到 https://<username>.github.io
```

### 常用调整速查
- 修改首页主标题/副标题：编辑 `index.markdown` 的 `hero_title`、`hero_subtitle`。
- 修改首页最近文章数量：在 `_layouts/home.html` 中调整 `{% for post in site.posts limit:3 %}` 的 `limit`。
- 调整导航：在 `_config.yml` 的 `navigation` 数组增删项，模板会自动渲染。
- 修改配色/间距/字体：编辑 `_layouts/home.html` 顶部 `<style>` 中的 CSS（同理可在 `blog.html`、`post.html` 中调整对应样式）。
- 设置站点信息：在 `_config.yml` 修改 `title`、`description`、`url`、`author`。

### 故障排查
- 本地运行报依赖错误：
  - 确认已安装 Ruby、Bundler；执行 `bundle install`。
- 本地样式不更新：
  - 清缓存或在开发时使用 `bundle exec jekyll serve --livereload`。
- GitHub Pages 未更新：
  - 检查仓库名是否为 `haoming58.github.io`；等待构建完成；查看仓库的 Pages 设置与构建日志。
- 中文路径/文件名问题：
  - 尽量使用英文文件名与路径，避免特殊字符导致的构建异常。

---

如果你希望加入「深色模式」、自定义字体或组件化片段（如页脚社交链接、文章目录 TOC），告诉我你的偏好，我可以直接帮你补齐对应的布局与样式，并写好注释，之后你可自行按注释修改。