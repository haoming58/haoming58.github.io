---
layout: page
title: notes
permalink: /notes/
description: Personal notes (no dates), quick references and snippets.
nav: true
nav_order: 4
---

<div class="notes">
  <ul>
    {% assign items = site.notes %}
    {% for note in items %}
    <li>
      <a href="{{ note.url | relative_url }}">{{ note.title }}</a>
      {% if note.description %}<span> — {{ note.description }}</span>{% endif %}
    </li>
    {% endfor %}
  </ul>
</div>



