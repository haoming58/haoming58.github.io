# Publications 链接和图片添加指南

## 📚 概述

本指南将教您如何在 publications 页面中添加 PDF、Website 链接和预览图片，让您的学术成果展示丰富而专业。

## 🖼️ 添加图片

### 1. 图片存放位置
- **预览图片**: `assets/img/publication_preview/`
- **支持格式**: PNG, JPG, GIF
- **推荐尺寸**: 200-400px 宽，150-300px 高

### 2. 在 BibTeX 中添加图片
在 `_bibliography/papers.bib` 文件中添加 `preview` 字段：

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  year={2024},
  preview={your_image_name.png},  # 添加预览图片
  pdf={/assets/pdf/your_paper.pdf},
  website={https://your-project.com}
}
```

### 3. 图片命名建议
- `antenna_design.png` - 天线设计相关
- `5g_antenna.png` - 5G天线相关
- `iot_disaster.png` - IoT灾害相关
- `machine_learning.png` - 机器学习相关

## 🔗 添加链接

### 1. 支持的链接类型

#### PDF 链接
```bibtex
pdf={your_paper.pdf}  # 本地PDF文件（系统会自动添加/assets/pdf/前缀）
pdf={https://example.com/paper.pdf}  # 外部PDF链接
```

#### 网站链接
```bibtex
website={https://your-project-website.com}
```

### 2. 完整示例

```bibtex
@article{example2024,
  title={Example Paper Title},
  author={Your Name and Co-author},
  year={2024},
  journal={Example Journal},
  
  % 图片
  preview={example_image.png},
  
  % 链接
  pdf={example2024.pdf},
  website={https://your-project.com}
}
```

## 📁 文件结构

```
assets/
├── img/
│   └── publication_preview/     # 论文预览图片
│       ├── antenna_design.png
│       ├── 5g_antenna.png
│       └── iot_disaster.png
└── pdf/                         # PDF文件
    ├── Design_of_Broadband_Microstrip_Quasi-Yagi_Antenna_with_Double_Branch_Structure.pdf
    ├── FullText.pdf
    └── The_Social_Impact_of_IoT_in_Disasters.pdf
```

## 🎨 显示效果

添加这些字段后，您的 publications 页面将显示：

1. **预览图片** - 在论文标题旁边显示
2. **PDF 按钮** - 点击下载或查看PDF文件
3. **Website 按钮** - 点击访问项目网站
4. **专业界面** - 图片和链接的完美结合

## 🚀 快速开始

1. **添加图片**: 将图片文件放入 `assets/img/publication_preview/`
2. **添加PDF**: 将PDF文件放入 `assets/pdf/`
3. **更新BibTeX**: 在 `_bibliography/papers.bib` 中添加 `preview`、`pdf` 和 `website` 字段
4. **测试**: 运行 `bundle exec jekyll serve` 查看效果

## 💡 提示

- 图片文件名要与 BibTeX 中的 `preview` 字段匹配
- PDF 文件路径只需要文件名，系统会自动添加 `/assets/pdf/` 前缀
- Website 链接直接使用完整URL
- 所有字段都是可选的，只添加您有的资源
