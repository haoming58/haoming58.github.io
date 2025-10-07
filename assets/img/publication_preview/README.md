# Publication Preview Images

This directory contains preview images for publications.

## How to Add Images

1. **Place your images here**: Put your publication preview images in this directory
2. **Supported formats**: PNG, JPG, GIF are recommended
3. **Naming convention**: Use descriptive names like `antenna_design.png`, `5g_antenna.png`, etc.
4. **Size recommendations**: 
   - Width: 200-400px
   - Height: 150-300px
   - Aspect ratio: 4:3 or 16:9 works well

## Example Images Needed

Based on your current publications, you should add:

- `antenna_design.png` - For the broadband microstrip antenna paper
- `5g_antenna.png` - For the 5G smartphone antenna paper  
- `iot_disaster.png` - For the IoT in disasters paper

## How to Reference in BibTeX

In your `_bibliography/papers.bib` file, add the `preview` field:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  year={2024},
  preview={your_image_name.png},  # This line adds the preview image
  pdf={/assets/pdf/your_paper.pdf},
  website={https://your-project.com}
}
```

The image will automatically appear in the publications page!
