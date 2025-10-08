# 本地使用指南（Jekyll · 无 Docker）

本指南帮助你在本地快速预览与维护本站点，并说明常见内容编辑方式。

## 1. 环境准备（Windows）

- 安装 Ruby+Devkit（推荐 RubyInstaller）：https://rubyinstaller.org/
- 安装 Bundler（若未自动安装）：
  ```powershell
  gem install bundler --no-document
  ```

## 2. 安装依赖与启动本地预览

```powershell
cd D:\Github\al-folio-main
bundle install
bundle exec jekyll serve --livereload --host 127.0.0.1 --port 4000
```

打开浏览器访问：`http://127.0.0.1:4000`

提示：已禁用上传/远程部署逻辑，仅做本地预览。保存文件会自动热更新。

## 3. 站点结构与你需要编辑的内容

- 站点配置：`_config.yml`
  - 基本信息：`title`, `first_name`, `middle_name`, `last_name`, `description`, `icon`
  - 本地预览 URL：`url: http://127.0.0.1:4000`, `baseurl: ""`
  - 可选特性：`enable_darkmode`, `enable_masonry` 等（true/false）
- 页面导航（已精简，仅保留以下页面）：`_pages/`
  - `about.md`（首页 `/`）
  - `blog.md`（博客 `/blog/`）
  - `publications.md`（论文 `/publications/`）
  - `notes.md`（笔记 `/notes/`）
  - `404.md`
  - 调整导航顺序：编辑各页面 Front Matter 的 `nav_order`
- 社交/联系方式：`_data/socials.yml`
  - 填写 `email:`，或 GitHub/LinkedIn 等；不需要的留空即可
  - 已移除他人 Google Scholar 等占位信息
- 头像与图片：`assets/img/`
  - 将头像放入该目录，在 `about.md` 中设置 `profile.image: 你的文件名`

## 4. 写博客（有时间线）

在 `_posts/` 新建文件（示例）：

```
_posts/2025-09-29-hello-world.md
```

内容模板：

```markdown
---
layout: post
title: 你的标题
description: 简短描述
categories: [分类A]
tags: [标签A, 标签B]
---
正文内容...
```

发布后访问：`/blog/`

## 5. 写笔记（无时间线，仅记录）

已创建 `notes` 集合与索引页：`/notes/`

- 新建笔记文件（示例）：`_notes/my-note.md`

```markdown
---
layout: page
title: 笔记主题
description: 可选
permalink: /notes/my-note/
---
笔记内容...
```

## 6. 维护论文列表（Publications）

- 编辑 `/_bibliography/papers.bib`（BibTeX）
- 页面：`/publications/`

## 7. 常见问题（Windows）

- PowerShell 命令行参数/引号报错：将长命令拆成多条依次执行
- 字体/图标或 CSS 警告：为上游依赖的提示，不影响本地预览
- 若端口被占用：修改 `--port` 或关闭占用进程
- 若监听变更不生效：在 `Gemfile` 中按提示加入 `gem 'wdm', '>= 0.1.0'` 后 `bundle install`

## 8. 关闭/启用特性（可选）

在 `_config.yml` 调整：

- 主题功能：`enable_darkmode`, `enable_masonry`, `enable_math`, `enable_progressbar` 等
- 搜索相关：已关闭 `search_enabled` 及相关脚本；如需启用，反向设为 true 并恢复 `assets/js/search/`
- 图片处理：已禁用 `jekyll-imagemagick`，如需启用请先安装 ImageMagick 并设 `imagemagick.enabled: true`

## 9. 备份与还原

- 本地仅预览，不会推送到 GitHub
- 若需备份，复制整个文件夹或自行初始化新的 Git 仓库（注意当前 `.git` 可能已被禁用/重命名）

## 10. 我需要做什么？（最小步骤）

1) 改 `_config.yml` 的个人信息
2) 放头像到 `assets/img/` 并在 `about.md` 设置 `profile.image`
3) 写一篇 `_posts/` 博客与一条 `_notes/` 笔记
4) 运行本地预览，访问四个页面：`/`、`/blog/`、`/publications/`、`/notes/`

如需我代填你的 About、示例博客与笔记，请把你的姓名、头像文件名、个人简介与邮箱发我。
