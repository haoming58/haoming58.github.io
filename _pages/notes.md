---
layout: note_with_toc
title: 学习笔记
permalink: /notes/
description: 个人学习笔记，包含机器学习和天线设计等各类知识总结
nav: true
nav_order: 4
---

<div class="notes">
  {% comment %} Generate categories overview {% endcomment %}
  {% assign categories = site.notes | group_by: 'category' | sort: 'name' %}
  {% assign uncategorized = site.notes | where: 'category', blank %}
  
  {% if categories.size > 0 or uncategorized.size > 0 %}
    <div class="notes-toc" id="categories">
      <h2 class="toc-title">📚 Note Categories</h2>
      <div class="toc-content">
        {% for category in categories %}
          {% if category.name != blank %}
            <div class="toc-category">
              <a href="#" class="toc-link" data-category="{{ category.name | slugify }}">
                <span class="toc-category-name">{{ category.name }}</span>
                <span class="toc-count">({{ category.items.size }} notes)</span>
              </a>
            </div>
          {% endif %}
        {% endfor %}
        {% if uncategorized.size > 0 %}
          <div class="toc-category">
            <a href="#" class="toc-link" data-category="uncategorized">
              <span class="toc-category-name">Uncategorized</span>
              <span class="toc-count">({{ uncategorized.size }} notes)</span>
            </a>
          </div>
        {% endif %}
      </div>
    </div>
  {% endif %}

  {% comment %} Organize notes by category {% endcomment %}
  {% for category in categories %}
    {% if category.name != blank %}
      <div class="category-section" id="{{ category.name | slugify }}" style="display: none;">
        <div class="notes-header">
          <h1>{{ category.name }} 笔记</h1>
          <p class="notes-description">{{ category.name }}相关的学习笔记和教程</p>
          <a href="#" class="back-link" onclick="showCategories()">← 返回分类列表</a>
        </div>
        <ul class="notes-list">
          {% for note in category.items %}
          <li class="note-item">
            <a href="{{ note.url | relative_url }}" onclick="storeCurrentCategory('{{ category.name | slugify }}')">{{ note.title }}</a>
            {% if note.description %}<span class="note-description"> — {{ note.description }}</span>{% endif %}
            {% if note.tags %}
              <div class="note-tags">
                {% for tag in note.tags %}
                  <span class="tag">{{ tag }}</span>
                {% endfor %}
              </div>
            {% endif %}
          </li>
          {% endfor %}
        </ul>
      </div>
    {% endif %}
  {% endfor %}
  
  {% comment %} Show uncategorized notes {% endcomment %}
  {% if uncategorized.size > 0 %}
    <div class="category-section" id="uncategorized" style="display: none;">
      <div class="notes-header">
        <h1>未分类笔记</h1>
        <p class="notes-description">没有特定分类的学习笔记</p>
        <a href="#" class="back-link" onclick="showCategories()">← Back to Categories</a>
      </div>
      <ul class="notes-list">
        {% for note in uncategorized %}
        <li class="note-item">
          <a href="{{ note.url | relative_url }}">{{ note.title }}</a>
          {% if note.description %}<span class="note-description"> — {{ note.description }}</span>{% endif %}
          {% if note.tags %}
            <div class="note-tags">
              {% for tag in note.tags %}
                <span class="tag">{{ tag }}</span>
              {% endfor %}
            </div>
          {% endif %}
        </li>
        {% endfor %}
      </ul>
    </div>
  {% endif %}
</div>

<script>
function showCategory(categoryId) {
  // Hide all category sections
  const sections = document.querySelectorAll('.category-section');
  sections.forEach(section => {
    section.style.display = 'none';
  });
  
  // Hide categories overview
  const toc = document.querySelector('.notes-toc');
  if (toc) {
    toc.style.display = 'none';
  }
  
  // Show selected category
  const categorySection = document.getElementById(categoryId);
  if (categorySection) {
    categorySection.style.display = 'block';
    // Update TOC for this category
    updateCategoryTOC(categoryId);
  }
}

function showCategories() {
  // Hide all category sections
  const sections = document.querySelectorAll('.category-section');
  sections.forEach(section => {
    section.style.display = 'none';
  });
  
  // Show categories overview
  const toc = document.querySelector('.notes-toc');
  if (toc) {
    toc.style.display = 'block';
  }
  
  // Reset TOC to categories view
  generateNotesTOC();
}

// Add click event listeners to category links
document.addEventListener('DOMContentLoaded', function() {
  const categoryLinks = document.querySelectorAll('.toc-link[data-category]');
  categoryLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      const categoryId = this.getAttribute('data-category');
      showCategory(categoryId);
    });
  });
});

// Custom TOC for notes page
function generateNotesTOC() {
  const tocContent = document.getElementById('toc-content');
  if (!tocContent) return;
  
  // Generate TOC based on the notes structure
  let tocHTML = '<ul>';
  
  // Add main sections
  tocHTML += '<li><a href="#categories" class="toc-level-1">📚 笔记分类</a></li>';
  
  // Add category links
  const categoryLinks = document.querySelectorAll('.toc-link[data-category]');
  categoryLinks.forEach(link => {
    const categoryName = link.querySelector('.toc-category-name').textContent;
    const count = link.querySelector('.toc-count').textContent;
    tocHTML += `<li><a href="#" class="toc-level-2" data-category="${link.getAttribute('data-category')}">${categoryName} ${count}</a></li>`;
  });
  
  tocHTML += '</ul>';
  tocContent.innerHTML = tocHTML;
  
  // Add click handlers for category links
  const tocCategoryLinks = tocContent.querySelectorAll('a[data-category]');
  tocCategoryLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      const categoryId = this.getAttribute('data-category');
      showCategory(categoryId);
    });
  });
  
  // Add click handlers for anchor links
  const tocAnchorLinks = tocContent.querySelectorAll('a[href^="#"]');
  tocAnchorLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      const targetId = this.getAttribute('href').substring(1);
      const targetElement = document.getElementById(targetId);
      
      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
}

// Update TOC when category is shown
function updateCategoryTOC(categoryId) {
  const tocContent = document.getElementById('toc-content');
  if (!tocContent) return;
  
  let tocHTML = '<ul>';
  tocHTML += '<li><a href="#" class="toc-level-1" onclick="showCategories()">← 返回分类列表</a></li>';
  
  // Add notes in this category
  const categorySection = document.getElementById(categoryId);
  if (categorySection) {
    const noteItems = categorySection.querySelectorAll('.note-item a');
    noteItems.forEach((link, index) => {
      const title = link.textContent.trim();
      const href = link.getAttribute('href');
      tocHTML += `<li><a href="${href}" class="toc-level-2">${title}</a></li>`;
    });
  }
  
  tocHTML += '</ul>';
  tocContent.innerHTML = tocHTML;
}

// Function to store current category when clicking on a note
function storeCurrentCategory(categoryId) {
  sessionStorage.setItem('currentCategory', categoryId);
}

// Generate TOC when page loads
document.addEventListener('DOMContentLoaded', function() {
  generateNotesTOC();
  
  // Check if we should show a specific category
  const currentCategory = sessionStorage.getItem('currentCategory');
  if (currentCategory) {
    // Show the specific category
    showCategory(currentCategory);
    // Clear the stored category
    sessionStorage.removeItem('currentCategory');
  }
});
</script>

<style>
/* Table of contents styles */
.notes-toc {
  background: var(--global-bg-color);
  border: 1px solid var(--global-border-color);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 2rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.toc-title {
  color: var(--global-theme-color);
  margin-bottom: 1rem;
  font-size: 1.5rem;
  text-align: center;
}

.toc-content {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.toc-category {
  text-align: center;
}

.toc-link {
  display: block;
  padding: 0.75rem 1rem;
  background: var(--global-bg-color);
  border: 2px solid var(--global-border-color);
  border-radius: 8px;
  text-decoration: none;
  color: var(--global-text-color);
  transition: all 0.3s ease;
  font-weight: 500;
}

.toc-link:hover {
  background: var(--global-theme-color);
  color: white;
  border-color: var(--global-theme-color);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.toc-category-name {
  display: block;
  font-size: 1.1rem;
  margin-bottom: 0.25rem;
}

.toc-count {
  font-size: 0.9rem;
  opacity: 0.8;
}

/* Category title styles */
.category-title {
  color: var(--global-theme-color);
  border-bottom: 2px solid var(--global-theme-color);
  padding-bottom: 0.5rem;
  margin-top: 2rem;
  margin-bottom: 1rem;
  scroll-margin-top: 2rem; /* Provide offset for anchor links */
}

/* Notes list styles */
.notes-list {
  list-style: none;
  padding-left: 0;
}

.note-item {
  margin-bottom: 1rem;
  padding: 0.75rem;
  background: var(--global-bg-color);
  border: 1px solid var(--global-border-color);
  border-radius: 8px;
  transition: all 0.3s ease;
}

.note-item:hover {
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  transform: translateY(-2px);
}

.note-item a {
  color: var(--global-text-color);
  text-decoration: none;
  font-weight: 500;
}

.note-item a:hover {
  color: var(--global-theme-color);
}

.note-description {
  color: var(--global-text-color-light);
  font-size: 0.9rem;
}

.note-tags {
  margin-top: 0.5rem;
}

.tag {
  display: inline-block;
  background: var(--global-theme-color);
  color: white;
  padding: 0.2rem 0.5rem;
  border-radius: 12px;
  font-size: 0.8rem;
  margin-right: 0.5rem;
  margin-bottom: 0.25rem;
}

/* Notes header styles */
.notes-header {
  text-align: center;
  margin-bottom: 2rem;
}

.notes-header h1 {
  color: var(--global-theme-color);
  margin-bottom: 0.5rem;
}

.notes-description {
  color: var(--global-text-color-light);
  font-size: 1.1rem;
  margin-bottom: 1rem;
}

.back-link {
  display: inline-block;
  color: var(--global-theme-color);
  text-decoration: none;
  font-weight: 500;
  padding: 0.5rem 1rem;
  border: 2px solid var(--global-theme-color);
  border-radius: 8px;
  transition: all 0.3s ease;
}

.back-link:hover {
  background: var(--global-theme-color);
  color: white;
}

.no-notes {
  text-align: center;
  padding: 2rem;
  color: var(--global-text-color-light);
}

/* Responsive design */
@media (max-width: 768px) {
  .toc-content {
    grid-template-columns: 1fr;
  }
  
  .toc-link {
    padding: 0.5rem;
  }
  
  .toc-category-name {
    font-size: 1rem;
  }
}
</style>



