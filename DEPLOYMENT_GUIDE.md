# GitHub Pages 部署指南与问题排查记录

## 目录

1. [项目概述](#项目概述)
2. [技术栈](#技术栈)
3. [项目运行原理](#项目运行原理)
4. [部署流程](#部署流程)
5. [问题排查记录](#问题排查记录)
6. [常见问题与解决方案](#常见问题与解决方案)

---

## 项目概述

这是一个基于 Jekyll 的学术个人网站，托管在 GitHub Pages 上。网站包含：
- 个人简介
- 出版物列表
- 博客文章
- 学习笔记（支持数学公式、目录导航）

---

## 技术栈

### 核心技术
- **Jekyll 4.4.1**: 静态网站生成器
- **Ruby 3.2**: 运行环境
- **GitHub Actions**: CI/CD 自动化部署
- **GitHub Pages**: 网站托管服务

### 关键依赖
```ruby
# Gemfile
gem 'jekyll'                          # 静态网站生成器
gem 'jekyll-sass-converter', '~> 3.0.0'  # Sass/SCSS 编译器
gem 'sass-embedded', '~> 1.77.8'      # Sass 编译引擎

# Jekyll 插件
gem 'jekyll-archives-v2'              # 文章归档
gem 'jekyll-feed'                     # RSS 订阅
gem 'jekyll-redirect-from'            # URL 重定向
gem 'jekyll-sitemap'                  # 站点地图
gem 'jemoji'                          # Emoji 支持
```

---

## 项目运行原理

### 1. 本地开发流程

```bash
# 1. 安装依赖
bundle install

# 2. 本地预览（开发模式）
bundle exec jekyll serve
# 访问 http://localhost:4000

# 3. 构建静态文件（生产模式）
JEKYLL_ENV=production bundle exec jekyll build
# 输出到 _site/ 目录
```

### 2. 文件结构

```
haoming58.github.io/
├── _config.yml              # Jekyll 配置文件
├── _layouts/                # 页面布局模板
│   ├── default.html
│   ├── note_with_toc.html   # 笔记页面（带目录）
│   └── ...
├── _includes/               # 可复用组件
├── _sass/                   # Sass/SCSS 样式文件
│   ├── _themes.scss         # 主题配置
│   └── ...
├── _notes/                  # 学习笔记 Markdown 文件
│   └── deep-learning/
│       └── rnn/
│           └── 06 - 通过时间反向传播.md
├── assets/                  # 静态资源
│   ├── css/
│   ├── js/
│   └── img/
├── .github/workflows/       # GitHub Actions 工作流
│   └── deploy.yml
├── Gemfile                  # Ruby 依赖声明
└── Gemfile.lock             # 依赖版本锁定
```

### 3. 笔记页面工作原理

#### Front Matter 配置
```yaml
---
layout: note_with_toc           # 使用带目录的布局
title: 6 通过时间反向传播
description: BPTT算法详解
category: Machine Learning
tags: [RNN, Deep Learning, BPTT]
permalink: /notes/backpropagation-through-time/  # 英文 URL
redirect_from:
  - /notes/通过时间反向传播/    # 旧的中文 URL 自动重定向
---
```

#### 关键特性
- **自动生成目录**：JavaScript 解析 `h1`-`h6` 标题
- **数学公式支持**：MathJax 渲染 LaTeX 公式
- **URL 重定向**：`jekyll-redirect-from` 插件处理旧链接
- **响应式布局**：移动端友好的侧边栏

---

## 部署流程

### 方式一：GitHub Actions 自动部署（推荐）

#### 1. GitHub 仓库设置
1. 进入仓库 **Settings** → **Pages**
2. 设置 **Source** 为 `GitHub Actions`
3. 保存配置

#### 2. 工作流配置（`.github/workflows/deploy.yml`）

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: ["main"]      # main 分支有推送时触发
  workflow_dispatch:        # 支持手动触发

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Setup Ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: '3.2'
          bundler-cache: true    # 自动缓存依赖
          
      - name: Setup Pages
        uses: actions/configure-pages@v4
        
      - name: Build with Jekyll
        run: bundle exec jekyll build --baseurl ""
        env:
          JEKYLL_ENV: production
          
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./_site

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        uses: actions/deploy-pages@v4
```

#### 3. 自动部署流程

```
本地提交 → git push → GitHub Actions 触发
    ↓
1. Checkout 代码
2. 安装 Ruby 3.2 和依赖
3. 构建 Jekyll 网站
4. 上传构建产物到 GitHub Pages
5. 部署到 https://haoming58.github.io
    ↓
部署完成（1-3 分钟）
```

### 方式二：本地构建 + 手动部署

```bash
# 1. 构建网站
JEKYLL_ENV=production bundle exec jekyll build

# 2. 推送到 gh-pages 分支
git subtree push --prefix _site origin gh-pages
```

---

## 问题排查记录

### 问题 1：文档内容不完整（Git 合并冲突）

#### 现象
- 文档结尾突然截断
- 出现 `<<<<<<<`, `=======`, `>>>>>>>` 标记
- GitHub 上显示格式错误

#### 根因
Git 合并时产生冲突，冲突标记未清理

#### 解决方案
```bash
# 1. 搜索冲突标记
grep -r "<<<<<<" .
grep -r "======" .
grep -r ">>>>>>" .

# 2. 手动编辑文件，删除冲突标记，保留正确内容
# 3. 提交修复
git add .
git commit -m "resolve merge conflicts"
git push
```

---

### 问题 2：Permalink 导致 404

#### 现象
- 旧的中文 URL `/notes/通过时间反向传播/` 可访问
- 新的英文 URL `/notes/backpropagation-through-time/` 返回 404

#### 根因
Front matter 配置错误或未生效

#### 解决方案
修改 `06 - 通过时间反向传播.md` 的 Front matter：

```yaml
---
layout: note_with_toc
title: 6 通过时间反向传播
permalink: /notes/backpropagation-through-time/  # 设置英文 URL
redirect_from:
  - /notes/通过时间反向传播/                    # 旧 URL 重定向
---
```

**注意事项**：
- `permalink` 必须以 `/` 开头和结尾
- 确保 `_config.yml` 中启用了 `jekyll-redirect-from` 插件
- 重新构建后需要等待 1-3 分钟生效

---

### 问题 3：GitHub Actions 构建失败（sass-embedded）

#### 现象
```
Error: Could not find gem 'sass-embedded (= 1.77.6)'
```

#### 根因
`sass-embedded` 版本 1.77.6 在 RubyGems 上**不存在**（版本号跳过）

#### 解决方案 1（失败）
```ruby
# Gemfile
gem 'sass-embedded', '~> 1.69.0'
```
**结果**：安装成功，但构建失败（不支持 `color.channel()` 函数）

#### 解决方案 2（失败）
```ruby
# Gemfile
gem 'sass-embedded', '~> 1.77.8'
```
**结果**：安装成功，但构建失败（仍不支持 `color.channel()` 函数）

#### 最终解决方案（成功）
修改 `_sass/_themes.scss`，替换新 API 为传统 API：

```scss
/* 旧代码（不兼容） */
--global-back-to-top-bg-color: rgba(
  #{color.channel($black-color, "red", $space: rgb)},
  #{color.channel($black-color, "green", $space: rgb)},
  #{color.channel($black-color, "blue", $space: rgb)},
  0.4
);

/* 新代码（兼容所有版本） */
--global-back-to-top-bg-color: rgba(
  #{red($black-color)},
  #{green($black-color)},
  #{blue($black-color)},
  0.4
);
```

**原因**：
- `color.channel()` 是 Sass Color Module Level 4 的新 API
- `sass-embedded 1.77.8` 及更早版本均不支持
- `red()`, `green()`, `blue()` 是 Sass 1.0 就存在的传统函数，完全兼容

---

### 问题 4：bundle install 失败（依赖冲突）

#### 现象
```
Error: This Bundle hasn't been installed yet. 
Run `bundle install` to update and install the bundled gems.
```

#### 根因
工作流中先执行 `bundle update`，但依赖尚未安装

#### 解决方案
修改 `.github/workflows/deploy.yml`：

```yaml
# 错误写法
- name: Install dependencies
  run: |
    bundle update jekyll-sass-converter sass-embedded
    bundle install

# 正确写法
- name: Setup Ruby
  uses: ruby/setup-ruby@v1
  with:
    ruby-version: '3.2'
    bundler-cache: true  # 自动处理依赖安装和缓存
```

---

## 常见问题与解决方案

### 1. 网站未更新

**排查步骤**：
1. 检查 GitHub Actions 是否成功运行（绿色勾）
2. 查看最新一次运行的日志
3. 确认提交已推送到 `main` 分支
4. 清除浏览器缓存或使用隐身模式

**构建时间**：通常 1-3 分钟

---

### 2. 数学公式不显示

**检查项**：
1. `_config.yml` 中 `enable_math: true`
2. 页面 Front matter 中未设置 `math: false`
3. MathJax 脚本已加载

**正确的公式语法**：
```markdown
行内公式：$E = mc^2$
块级公式：
$$
\frac{\partial L}{\partial W_h} = \sum_{t=1}^T \frac{\partial \ell_t}{\partial h_t} \frac{\partial h_t}{\partial W_h}
$$
```

---

### 3. 目录不生成

**原因**：
- 文档中没有标题（`h1`-`h6`）
- JavaScript 执行失败

**解决方案**：
1. 确保文档有合适的标题结构
2. 检查浏览器控制台是否有 JavaScript 错误
3. 验证 `_layouts/note_with_toc.html` 中的 TOC 生成脚本

---

### 4. 样式丢失

**可能原因**：
- Sass/SCSS 编译失败
- `assets/css/main.scss` 路径错误
- `_config.yml` 中 `baseurl` 配置错误

**解决方案**：
```yaml
# _config.yml
baseurl: ""  # GitHub Pages 个人站点留空
url: "https://haoming58.github.io"
```

---

### 5. 中文 URL 编码问题

**推荐做法**：
- 使用英文 `permalink`
- 用 `redirect_from` 保留旧的中文 URL 兼容性

```yaml
permalink: /notes/backpropagation-through-time/
redirect_from:
  - /notes/通过时间反向传播/
  - /notes/deep-learning/rnn/06/
```

---

## 部署检查清单

### 提交前
- [ ] 本地 `bundle exec jekyll serve` 预览正常
- [ ] 所有 Markdown 文件 Front matter 正确
- [ ] 图片链接使用相对路径或 `{{ site.baseurl }}`
- [ ] 无 Git 合并冲突标记

### 推送后
- [ ] GitHub Actions 构建成功（绿色勾）
- [ ] 访问 https://haoming58.github.io 验证
- [ ] 检查 `sitemap.xml` 是否包含新页面
- [ ] 测试新旧 URL 是否都能访问（redirect_from）

---

## 参考资源

- [Jekyll 官方文档](https://jekyllrb.com/docs/)
- [GitHub Pages 文档](https://docs.github.com/en/pages)
- [Sass 官方文档](https://sass-lang.com/documentation)
- [MathJax 文档](https://docs.mathjax.org/)
- [jekyll-redirect-from 插件](https://github.com/jekyll/jekyll-redirect-from)

---

## 版本历史

| 日期 | 版本 | 说明 |
|------|------|------|
| 2025-11-04 | 1.0 | 初始版本，记录 BPTT 文档部署问题排查过程 |

---

## 联系方式

如有问题，请在 [GitHub Issues](https://github.com/haoming58/haoming58.github.io/issues) 中提出。
