---
layout: note_with_toc
title: 学习笔记
permalink: /notes/
description: 个人学习笔记，包含机器学习和天线设计等各类知识总结
nav: true
nav_order: 3
---

<div class="notes">
  {% comment %} Generate categories overview {% endcomment %}
  {% assign categories = site.notes | group_by: 'category' | sort: 'name' %}
  {% assign uncategorized = site.notes | where: 'category', blank %}

  {% if categories.size > 0 or uncategorized.size > 0 %}
    <div class="notes-toc" id="categories">
      <div class="notes-overview-header">
        <h1>学习笔记</h1>
        <p class="notes-overview-sub">{{ site.notes.size }} 篇笔记 · {{ categories.size }} 个分类</p>
      </div>
      <div class="categories-grid">
        {% for category in categories %}
          {% if category.name != blank %}
            {% assign subs = category.items | group_by: 'subcategory' | sort: 'name' %}
            <a href="#" class="category-card toc-link" data-category="{{ category.name | slugify }}">
              <div class="category-card-header">
                <h3 class="category-card-title">{{ category.name }}</h3>
                <span class="category-card-count">{{ category.items.size }}</span>
              </div>
              <div class="category-card-subs">
                {% for sub in subs %}
                  {% if sub.name != blank and sub.name != "" %}
                    <span class="sub-chip">{{ sub.name }} <em>{{ sub.items.size }}</em></span>
                  {% endif %}
                {% endfor %}
                {% assign uncat_in_cat = category.items | where: 'subcategory', blank %}
                {% if uncat_in_cat.size > 0 %}
                  <span class="sub-chip sub-chip-misc">其他 <em>{{ uncat_in_cat.size }}</em></span>
                {% endif %}
              </div>
            </a>
          {% endif %}
        {% endfor %}
        {% if uncategorized.size > 0 %}
          <a href="#" class="category-card toc-link" data-category="uncategorized">
            <div class="category-card-header">
              <h3 class="category-card-title">Uncategorized</h3>
              <span class="category-card-count">{{ uncategorized.size }}</span>
            </div>
            <div class="category-card-subs">
              <span class="sub-chip sub-chip-misc">未分类</span>
            </div>
          </a>
        {% endif %}
      </div>
    </div>
  {% endif %}

  {% comment %} Organize notes by category {% endcomment %}
  {% for category in categories %}
    {% if category.name != blank %}
      <div class="category-section" id="{{ category.name | slugify }}" style="display: none;">
        <div class="notes-header">
          <a href="#" class="back-link" onclick="showCategories()">← 返回分类</a>
          <h1>{{ category.name }}</h1>
          <p class="notes-description">{{ category.items.size }} 篇笔记</p>
        </div>

        {% comment %} Group by subcategory if exists {% endcomment %}
        {% assign subcategories = category.items | group_by: 'subcategory' | sort: 'name' %}
        {% assign has_subcategories = false %}
        {% for subcategory in subcategories %}
          {% if subcategory.name != blank and subcategory.name != "" %}
            {% assign has_subcategories = true %}
            {% break %}
          {% endif %}
        {% endfor %}

        {% if has_subcategories %}
          <nav class="subcategory-nav">
            {% for subcategory in subcategories %}
              {% if subcategory.name != blank and subcategory.name != "" %}
                <a href="#sub-{{ subcategory.name | slugify }}" class="subcat-tab">
                  {{ subcategory.name }} <em>{{ subcategory.items.size }}</em>
                </a>
              {% endif %}
            {% endfor %}
          </nav>

          {% for subcategory in subcategories %}
            {% if subcategory.name != blank and subcategory.name != "" %}
              <h2 class="subcategory-title" id="sub-{{ subcategory.name | slugify }}">{{ subcategory.name }}</h2>
              <ul class="notes-list">
                {% for note in subcategory.items %}
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
            {% endif %}
          {% endfor %}
        {% else %}
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
        {% endif %}
      </div>
    {% endif %}
  {% endfor %}

  {% comment %} Show uncategorized notes {% endcomment %}
  {% if uncategorized.size > 0 %}
    <div class="category-section" id="uncategorized" style="display: none;">
      <div class="notes-header">
        <a href="#" class="back-link" onclick="showCategories()">← 返回分类</a>
        <h1>未分类笔记</h1>
        <p class="notes-description">{{ uncategorized.size }} 篇笔记</p>
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
  tocHTML += '<li><a href="#categories" class="toc-level-1">笔记分类</a></li>';
  
  // Add category links
  const categoryLinks = document.querySelectorAll('.toc-link[data-category]');
  categoryLinks.forEach(link => {
    const titleEl = link.querySelector('.category-card-title');
    const countEl = link.querySelector('.category-card-count');
    const categoryName = titleEl ? titleEl.textContent.trim() : link.getAttribute('data-category');
    const count = countEl ? `(${countEl.textContent.trim()})` : '';
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
/* ============================================================
   Notes index — Notion-style card grid
   Uses theme CSS variables so dark/light switching just works.
   ============================================================ */

/* Overview header */
.notes-overview-header {
  margin-bottom: 2rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid var(--global-divider-color);
}

.notes-overview-header h1 {
  margin: 0 0 0.35rem 0;
  font-size: 2rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--global-text-color);
}

.notes-overview-sub {
  margin: 0;
  color: var(--global-text-color-light);
  font-size: 0.95rem;
}

/* Categories grid */
.notes-toc {
  margin-bottom: 2.5rem;
}

.categories-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1rem;
}

.category-card {
  display: block;
  padding: 1.25rem 1.35rem;
  background: var(--global-card-bg-color, var(--global-bg-color));
  border: 1px solid var(--global-divider-color);
  border-radius: 10px;
  text-decoration: none;
  color: inherit;
  position: relative;
  overflow: hidden;
  transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
}

.category-card::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 3px;
  background: var(--global-theme-color);
  opacity: 0;
  transition: opacity 0.15s ease;
}

.category-card:hover {
  border-color: var(--global-theme-color);
  transform: translateY(-1px);
  box-shadow: 0 4px 16px var(--global-divider-color);
  text-decoration: none;
}

.category-card:hover::before {
  opacity: 1;
}

.category-card-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.85rem;
}

.category-card-title {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--global-text-color);
  letter-spacing: -0.01em;
}

.category-card-count {
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--global-text-color-light);
  background: var(--global-divider-color);
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  white-space: nowrap;
}

.category-card-subs {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.sub-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.8rem;
  padding: 0.2rem 0.6rem;
  border-radius: 5px;
  background: var(--global-divider-color);
  color: var(--global-text-color-light);
}

.sub-chip em {
  font-style: normal;
  font-size: 0.72rem;
  opacity: 0.7;
}

.sub-chip-misc {
  opacity: 0.7;
}

/* Subcategory sticky nav inside a category */
.subcategory-nav {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  padding: 0.85rem 0;
  margin-bottom: 0.5rem;
  background: var(--global-bg-color);
  border-bottom: 1px solid var(--global-divider-color);
}

.subcat-tab {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.85rem;
  padding: 0.35rem 0.75rem;
  border-radius: 6px;
  text-decoration: none;
  color: var(--global-text-color-light);
  background: transparent;
  border: 1px solid var(--global-divider-color);
  transition: all 0.15s ease;
}

.subcat-tab em {
  font-style: normal;
  font-size: 0.75rem;
  opacity: 0.7;
}

.subcat-tab:hover,
.subcat-tab.active {
  color: var(--global-theme-color);
  border-color: var(--global-theme-color);
  background: var(--global-card-bg-color, transparent);
  text-decoration: none;
}

/* Subcategory section title */
.subcategory-title {
  font-size: 1.4rem;
  font-weight: 600;
  margin-top: 2.5rem;
  margin-bottom: 1rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--global-divider-color);
  color: var(--global-text-color);
  scroll-margin-top: 5rem;
}

.subcategory-title:first-of-type {
  margin-top: 1.25rem;
}

/* Notes list */
.notes-list {
  list-style: none;
  padding-left: 0;
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  margin: 0;
}

.note-item {
  padding: 0.95rem 1.15rem;
  background: var(--global-card-bg-color, var(--global-bg-color));
  border: 1px solid var(--global-divider-color);
  border-radius: 8px;
  transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
  position: relative;
  overflow: hidden;
}

.note-item::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 2px;
  background: var(--global-theme-color);
  opacity: 0;
  transition: opacity 0.15s ease;
}

.note-item:hover {
  border-color: var(--global-theme-color);
  transform: translateY(-1px);
  box-shadow: 0 3px 12px var(--global-divider-color);
}

.note-item:hover::before {
  opacity: 1;
}

.note-item a {
  color: var(--global-text-color);
  text-decoration: none;
  font-weight: 500;
  font-size: 1rem;
  display: block;
  margin-bottom: 0.3rem;
  transition: color 0.15s ease;
}

.note-item a:hover {
  color: var(--global-theme-color);
  text-decoration: none;
}

.note-description {
  color: var(--global-text-color-light);
  font-size: 0.875rem;
  line-height: 1.55;
}

.note-tags {
  margin-top: 0.55rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}

.tag {
  display: inline-flex;
  align-items: center;
  color: var(--global-text-color-light);
  padding: 0.15rem 0.55rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 400;
  border: 1px solid var(--global-divider-color);
  transition: all 0.15s ease;
}

.tag:hover {
  background: var(--global-theme-color);
  color: var(--global-hover-text-color, #fff);
  border-color: var(--global-theme-color);
}

/* Category page header */
.notes-header {
  margin-bottom: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--global-divider-color);
}

.notes-header h1 {
  margin: 0.4rem 0 0.35rem 0;
  font-size: 1.85rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--global-text-color);
}

.notes-description {
  margin: 0;
  color: var(--global-text-color-light);
  font-size: 0.9rem;
}

.back-link {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.85rem;
  color: var(--global-text-color-light);
  text-decoration: none;
  transition: color 0.15s ease;
}

.back-link:hover {
  color: var(--global-theme-color);
  text-decoration: none;
}

/* Fade-in animation for cards */
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.category-card {
  animation: fadeIn 0.25s ease both;
}

.category-card:nth-child(1) { animation-delay: 0.02s; }
.category-card:nth-child(2) { animation-delay: 0.06s; }
.category-card:nth-child(3) { animation-delay: 0.10s; }
.category-card:nth-child(4) { animation-delay: 0.14s; }
.category-card:nth-child(5) { animation-delay: 0.18s; }
.category-card:nth-child(6) { animation-delay: 0.22s; }

/* Responsive */
@media (max-width: 768px) {
  .categories-grid {
    grid-template-columns: 1fr;
    gap: 0.85rem;
  }

  .notes-overview-header h1 {
    font-size: 1.55rem;
  }

  .notes-header h1 {
    font-size: 1.4rem;
  }

  .subcategory-nav {
    overflow-x: auto;
    flex-wrap: nowrap;
    margin-left: -0.5rem;
    margin-right: -0.5rem;
    padding-left: 0.5rem;
    padding-right: 0.5rem;
  }

  .subcat-tab {
    white-space: nowrap;
    flex-shrink: 0;
  }

  .note-item {
    padding: 0.85rem 1rem;
  }
}

/* Smooth scroll for anchor jumps */
html {
  scroll-behavior: smooth;
}
</style>



